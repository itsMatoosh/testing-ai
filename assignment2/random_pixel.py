import numpy as np
import json
from keras.utils import array_to_img, load_img, img_to_array
from matplotlib import pyplot as plt
rng = np.random.default_rng()


def mutate_random_pixels(seed, epsilon, n_pixels=1, n_neighbors=1):
    neighbors = []
    seed_idx = np.array([[i, j] for i in range(seed.shape[0])for j in range(seed.shape[1])])

    # print(seed_idx)
    for _ in range(n_neighbors):
        neighbor = seed.copy()

        # pixel coords

        idx = rng.choice(seed_idx, replace=False, size=n_pixels)
        pixels = neighbor[idx[:, 0], idx[:, 1]]
        pixels_mutated = np.array([mutate_pixel(pixel, epsilon) for pixel in pixels])
        neighbor[idx[:, 0], idx[:, 1]] = pixels_mutated

        neighbors.append(neighbor)

    return neighbors


def mutate_pixel(pixel, epsilon):
    # the euclidean distance between original and mutated pixel can not be e*255
    mutated_pixel = pixel.copy()
    distances = np.zeros(mutated_pixel.shape[0])
    total_allowed = 255 * epsilon
    for c in range(mutated_pixel.shape[0]):
        allowed_change = np.sqrt(total_allowed**2 - sum(distances**2))
        change = rng.integers(-allowed_change, allowed_change)
        mutated_pixel[c] = np.clip(mutated_pixel[c] + change, 0, 255)
        distances[c] = mutated_pixel[c] - pixel[c]

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

    out = mutate_random_pixels(seed, 0.9, n_pixels=100, n_neighbors=1)


    plt.imshow(img)
    plt.title("Original image")
    plt.show()

    for i, neighbor in enumerate(out):
        plt.imshow(array_to_img(neighbor))
        plt.title(f"Mutated image {i+1}")
        plt.show()
