import json
import os
import time
import numpy as np
import torch
from torchvision.utils import save_image

from keras.applications import vgg16

from keras.applications.imagenet_utils import decode_predictions
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image


from keras.utils import array_to_img, load_img, img_to_array
from hill_climbing import hill_climb

start_time = time.time()

# -----------------------------
# Utility: parse Imagenet prediction
# -----------------------------
def parse_prediction(output, categories):
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probs, 1)
    return categories[top_catid], top_prob.item()


# ================================================================
# 1. Load JSON file with images + expected human label
# ================================================================
JSON_FILE = "data/image_labels.json"
IMAGE_DIR = "images/"

with open(JSON_FILE, "r") as f:
    items = json.load(f)

# ================================================================
# 2. Load ImageNet labels
# ================================================================
with open("data/imagenet_classes.txt", "r") as f:
    imagenet_labels = [s.strip() for s in f.readlines()]

label_to_index = {label: i for i, label in enumerate(imagenet_labels)}


# ================================================================
# 3. Model
# ================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = vgg16.VGG16(weights="imagenet")

# ================================================================
# 5. Attack hyperparameters
# ================================================================
EPS = 0.30          # This can be tuned

# ================================================================
# 6. Output directory
# ================================================================
OUTDIR = "attack_results"
os.makedirs(OUTDIR, exist_ok=True)

# ================================================================
# 7. Run attacks for every image from the JSON file
# ================================================================
total_hc_time = 0.0

for entry in tqdm(items, desc="Running attacks"):
    image_file = entry["image"]
    human_label = entry["label"]  # e.g. "goldfish"

    # -----------------------------
    # Load + preprocess image
    # -----------------------------
    img = load_img(os.path.join(IMAGE_DIR, image_file))
    x = img_to_array(img)

    # -----------------------------
    # Ground truth index
    # -----------------------------
    if human_label in label_to_index:
        true_idx = label_to_index[human_label]
    else:
        true_idx = None
        print(f"⚠️ Warning: '{human_label}' not found in ImageNet labels.")

    # -----------------------------
    # Predict clean image
    # -----------------------------
    out_clean = model.predict(np.expand_dims(x, axis=0))
    _, pred_clean, prob_clean = decode_predictions(out_clean, top=1)[0][0]

    # Save clean image


    img = x.astype("float32")/255
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    save_image(img_tensor, os.path.join(OUTDIR, f"{image_file}_clean.png"))

    print(f"\nImage: {image_file}")
    print(f"Human label: {human_label}")
    print(f"Model prediction (clean): {pred_clean} ({prob_clean:.3f})")

    # =====================================================
    # hill climb attack Attack
    # =====================================================
    hc_start = time.time()
    x_hill, x_fit = hill_climb(x, model, target_label=human_label, epsilon=EPS)
    hc_end = time.time()
    total_hc_time += hc_end - hc_start

    out_hill = model.predict(np.expand_dims(x_hill, axis=0))
    # pred_hill, prob_hill = parse_prediction(out_fgm, imagenet_labels)

    _, pred_hill, prob_hill = decode_predictions(out_hill, top=1)[0][0]

    x_hill = x_hill.astype("float32")
    x_hill = x_hill/255
    img_tensor = torch.from_numpy(x_hill).permute(2, 0, 1)

    save_image(img_tensor, os.path.join(OUTDIR, f"{image_file}_hc.png"))

    print(f"HC prediction: {pred_hill} ({prob_hill:.3f})")

    # =====================================================
    # Summary for this image
    # =====================================================
    if true_idx is not None:
        print("\nCorrect label index:", true_idx)
        print("Clean correct?", imagenet_labels.index(pred_clean.replace("_", " ")) == true_idx)
        print("hc correct?", imagenet_labels.index(pred_hill.replace("_", " ")) == true_idx)

    print("------------------------------------------------------")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n======================================================")
print(f"Hill Climb attack time: {total_hc_time:.2f} seconds ({total_hc_time/60:.2f} minutes)")
print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"======================================================")