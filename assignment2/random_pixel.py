import numpy as np
import json
from keras.utils import array_to_img, load_img, img_to_array
from matplotlib import pyplot as plt
rng = np.random.default_rng()


def mutate_random_pixels(seed, epsilon, initial_img, n_pixels=1, n_neighbors=1):
    neighbors = []
    seed_idx = np.array([[i, j] for i in range(seed.shape[0])for j in range(seed.shape[1])])

    # print(seed_idx)
    for _ in range(n_neighbors):
        neighbor = seed.copy()

        # pixel coords

        idx = rng.choice(seed_idx, replace=False, size=n_pixels)
        pixels = neighbor[idx[:, 0], idx[:, 1]]
        initial_pixels = initial_img[idx[:, 0], idx[:, 1]]
        pixels_mutated = np.array([mutate_pixel(pixels[i], epsilon, initial_pixels[i]) for i in range(len(pixels))])
        neighbor[idx[:, 0], idx[:, 1]] = pixels_mutated

        neighbors.append(neighbor)

    return neighbors


def mutate_pixel(pixel, epsilon, initial_pxl):
    # the euclidean distance between original and mutated pixel can not be e*255
    mutated_pixel = pixel.copy()
    # calculate pre existing distance if the pixel was already change it will not be 0
    distances = mutated_pixel - initial_pxl
    total_allowed = np.array([255 * epsilon]*3)
    allowed_change = total_allowed - np.abs(distances)


    # in theory the upper and lower limit are not symmetric but whatever
    try:
        pixel_change = rng.integers(-allowed_change, allowed_change + 1)
    except ValueError as e:
        print("Allowed change:", allowed_change)
        print("Distances:", distances)
        print("Total allowed:", total_allowed)
        print("abs distances:", np.abs(distances))
        raise e


    mutated_pixel = mutated_pixel + pixel_change
    mutated_pixel = np.clip(mutated_pixel, 0, 255)
    return mutated_pixel


if __name__ == "__main__":

    # Load JSON describing dataset
    with open("data/image_labels.json") as f:
        image_list = json.load(f)

    # Pick first entry
    item = image_list[0]
    image_path = "images/" + item["image"]
    target_label = item["label"]

    print(f"Loaded image: {image_path}")
    print(f"Target label: {target_label}")

    img = load_img(image_path)

    img_array = img_to_array(img)
    seed = img_array.copy()

    out = mutate_random_pixels(seed, 0.1, seed, n_pixels=100, n_neighbors=1)

    for i, neighbor in enumerate(out):
        plt.imshow(array_to_img(neighbor))
        plt.title(f"Mutated image {i + 1}")
        plt.savefig(f"mutated_image_{i + 1}.png")
        plt.show()

    plt.imshow(img)
    plt.title("Original image")
    plt.show()
    for i in range(10):
        out = mutate_random_pixels(out[0], 0.1, seed, n_pixels=100, n_neighbors=1)




