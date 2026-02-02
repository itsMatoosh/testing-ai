import json
import pickle
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List, Tuple
from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import array_to_img, load_img, img_to_array

from random_pixel import mutate_random_pixels

from hill_climbing import hill_climb, compute_fitness, mutate_seed, select_best

plt.switch_backend('Tkagg')

def hill_climb_history(
        initial_seed: np.ndarray,
        model,
        target_label: str,
        epsilon: float = 0.30,
        iterations: int = 300
) -> Tuple[np.ndarray, float]:
    """
    Main hill-climbing loop.

    Requirements:
        ✓ Start from initial_seed
        ✓ EACH iteration:
              - Generate ANY number of neighbors using mutate_seed()
              - Enforce the SAME L∞ bound relative to initial_seed
              - Add current image to candidates (elitism)
              - Use select_best() to pick the winner
        ✓ Accept new candidate only if fitness improves
        ✓ Stop if:
              - target class is broken confidently, OR
              - no improvement for multiple steps (optional)

    """

    # Make a copy to avoid returning the original unchanged
    current_image = initial_seed.copy()
    current_fitness = compute_fitness(current_image, model, target_label)

    # Track the best perturbation found (even if attack doesn't fully succeed)
    best_ever_image = current_image.copy()
    best_ever_fitness = current_fitness

    stagnation_limit = 40
    stagnation_counter = 0

    # assignment defines succes as number 1 being nto correct, negative fitenss
    # so base threshold is 0, bigger negative is more restrictive
    threshold = 0
    fitness_history = [current_fitness]
    print(f"Initial fitness: {current_fitness:.4f}")

    for iteration in range(iterations):
        neighbors = mutate_seed(current_image, epsilon, initial_seed)
        candidates = neighbors + [current_image]  # Add current image for elitism

        best_image, best_fitness = select_best(candidates, model, target_label)
        fitness_history.append(best_fitness)

        if best_fitness < current_fitness:
            current_image = best_image.copy()
            current_fitness = best_fitness

            # Update best ever found
            if current_fitness < best_ever_fitness:
                best_ever_image = current_image.copy()
                best_ever_fitness = current_fitness

            stagnation_counter = 0

            print(f"Iteration {iteration + 1}: Improved fitness to {current_fitness:.4f}")
            if current_fitness < threshold:
                print(f"Target class broken confidently at iteration {iteration + 1}.")
                break

        else:
            # Even without improvement, check if any neighbor is better than best_ever
            if best_fitness < best_ever_fitness:
                best_ever_image = best_image.copy()
                best_ever_fitness = best_fitness

            stagnation_counter += 1
            if stagnation_counter >= stagnation_limit:
                print(f"Stopping early due to stagnation at iteration {iteration + 1}.")
                break
            print(f"Iteration {iteration + 1}: No improvement.")

    # Return the best perturbation found throughout the search
    print(f"Final best fitness: {best_ever_fitness:.4f}")

    return best_ever_image, best_ever_fitness, fitness_history


def run_n_hill_climb(n_tests=1, n_pictures=1):
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        model = vgg16.VGG16(weights="imagenet")
        with open('model.pkl', 'wb') as f:
            pickle.dump(vgg16.VGG16(weights="imagenet"), f)


    # Load JSON describing dataset
    with open("data/image_labels.json") as f:
        image_list = json.load(f)

    # Pick first entry
    out = {}
    for i in range(n_pictures):
        item = image_list[i]
        image_path = "images/" + item["image"]
        target_label = item["label"]

        print(f"Loaded image: {image_path}")
        print(f"Target label: {target_label}")

        img = load_img(image_path)
        # plt.imshow(img)
        # plt.title("Original image")
        # plt.show()

        img_array = img_to_array(img)
        seed = img_array.copy()

        # Print baseline top-5 predictions
        print("\nBaseline predictions (top-5):")
        preds = model.predict(np.expand_dims(seed, axis=0))
        for cl in decode_predictions(preds, top=5)[0]:
            print(f"{cl[1]:20s}  prob={cl[2]:.5f}")

        # time.sleep(1)
        all_histories = []
        for j in range(n_tests):
            print(f"\n--- Running hill climbing test {j+1}/{n_tests} on picture {i+1}/{n_pictures} ---")
        # Run hill climbing attack

            final_img, final_fitness, fitness_hist = hill_climb_history(
                initial_seed=seed,
                model=model,
                target_label=target_label,
                epsilon=0.30,
                iterations=300
            )
            all_histories.append(fitness_hist)
            print("\nFinal fitness:", final_fitness)

            # plt.imshow(array_to_img(final_img))
            # plt.title(f"Adversarial Result — fitness={final_fitness:.4f}")
            # plt.savefig(f"adversarial_result_{time.time()}.png")
            # plt.show()

            # Print final predictions
            final_preds = model.predict(np.expand_dims(final_img, axis=0))
            print("\nFinal predictions:")
            for cl in decode_predictions(final_preds, top=5)[0]:
                print(cl)
        out[f"picture_{i}"] = all_histories
    print(out)


def graph_history(fitness_history):

    for hist in fitness_history:
        plt.plot(hist[:-1])

    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Hill Climbing Fitness Over Iterations")
    plt.show()


def iteration_hist(fitness_history):
    iterations = [len(hist) for hist in fitness_history]
    plt.hist(iterations, bins=5)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Frequency")
    plt.title("Distribution of Iterations to Convergence")
    plt.show()


if __name__ == "__main__":
    #run_n_hill_climb(n_pictures=1, n_tests=10)
    # Example fitness history for graphing
    # hist = {'picture_0': [[0.9828675, 0.9812905, 0.97996706, 0.97555023, 0.97423935, 0.9703715, 0.96922183, 0.96509224, 0.96185803, 0.9582469, 0.9541826, 0.94958895, 0.94571966, 0.9403288, 0.93410736, 0.92875344, 0.9149777, 0.9069642, 0.9024018, 0.88907945, 0.8834411, 0.8671491, 0.8451379, 0.83338684, 0.8169887, 0.81164277, 0.8007252, 0.791876, 0.7136232, 0.6900176, 0.66932595, 0.6271342, 0.60677814, 0.5551248, 0.5320385, 0.5113089, 0.48043266, -0.47759062]]}
    hist = {'picture_0': [[0.9828675, 0.9811373, 0.9756835, 0.9735411, 0.9723786, 0.96474916, 0.96259135, 0.9591183, 0.9558864, 0.95231485, 0.9491564, 0.9408979, 0.9214495, 0.91506284, 0.90836126, 0.8997607, 0.8890129, 0.8778872, 0.86136234, 0.84725505, 0.82477677, 0.80381435, 0.7872119, 0.7753958, 0.7531735, 0.7204846, 0.6999253, 0.656458, 0.6405947, 0.60322785, 0.5761858, 0.5355776, 0.5163856, -0.48303735], [0.9828675, 0.9800187, 0.97651154, 0.974257, 0.9717861, 0.968031, 0.96617883, 0.96094996, 0.9579395, 0.95455194, 0.9498864, 0.94337285, 0.93645984, 0.9257106, 0.9175504, 0.9112339, 0.9059021, 0.8892122, 0.88500357, 0.8732788, 0.86513996, 0.849233, 0.8332569, 0.81171495, 0.80080616, 0.7822565, 0.760503, 0.75213605, 0.7348537, 0.7215409, 0.7109062, 0.68169284, 0.664141, 0.6251401, 0.6089989, 0.5946871, 0.5530947, 0.534754, 0.51526767, 0.4927608, -0.47922015], [0.9828675, 0.97987217, 0.9782899, 0.9762429, 0.9741887, 0.97015476, 0.9648586, 0.9612696, 0.9520224, 0.9474065, 0.93222904, 0.9280757, 0.9186191, 0.90973204, 0.896706, 0.88598615, 0.85639304, 0.84295994, 0.82387453, 0.8157058, 0.80406296, 0.75832057, 0.73981583, 0.7292558, 0.7119526, 0.6569603, 0.62243146, 0.602857, 0.581516, 0.55909884, 0.5262554, 0.4928436, -0.48310733], [0.9828675, 0.98078525, 0.9795116, 0.9777272, 0.9713131, 0.9671496, 0.9637415, 0.9591773, 0.9466617, 0.93932045, 0.9359, 0.928921, 0.91251224, 0.90256363, 0.8982233, 0.8883473, 0.87772983, 0.8702718, 0.8615454, 0.8564032, 0.8482118, 0.84129506, 0.80597174, 0.77714825, 0.7603728, 0.757176, 0.74065053, 0.7224065, 0.70727074, 0.6797924, 0.6586455, 0.6406051, 0.61865485, 0.60031104, 0.57620066, 0.5632887, 0.5200391, 0.49674746, -0.47567537], [0.9828675, 0.9818038, 0.97882587, 0.9751864, 0.97268564, 0.97131246, 0.96874744, 0.96616834, 0.95923084, 0.9556241, 0.9508183, 0.94219553, 0.92871666, 0.9239153, 0.91574234, 0.9103476, 0.90062463, 0.88689476, 0.8750585, 0.8634857, 0.8584827, 0.8159489, 0.7993303, 0.79212874, 0.7769628, 0.76348674, 0.74936163, 0.7299597, 0.71925783, 0.7015811, 0.6860415, 0.6726456, 0.65031433, 0.6401057, 0.61069727, 0.5761579, 0.5495212, 0.52959824, 0.52159756, -0.46775144], [0.9828675, 0.98123735, 0.9796469, 0.9777339, 0.9756246, 0.97366446, 0.97059536, 0.96895903, 0.9663521, 0.96400607, 0.95950866, 0.9563574, 0.9521105, 0.94846356, 0.9452647, 0.937179, 0.93165153, 0.9214335, 0.91911125, 0.91109544, 0.9003504, 0.8952241, 0.8870457, 0.8817658, 0.86904866, 0.8557809, 0.84884965, 0.8388784, 0.8172009, 0.801357, 0.78588116, 0.7723747, 0.7502837, 0.7406677, 0.72207296, 0.69025177, 0.6511224, 0.638062, 0.6251705, 0.5803689, 0.5431742, 0.52772397, -0.4600402], [0.9828675, 0.9788447, 0.9754254, 0.9706453, 0.9670113, 0.962025, 0.95887977, 0.9543861, 0.9464199, 0.94041955, 0.93249, 0.92889374, 0.9225558, 0.9103135, 0.89551204, 0.87875986, 0.8754867, 0.8600368, 0.84929883, 0.81418276, 0.78106457, 0.7710103, 0.74341, 0.72713286, 0.7069554, 0.6662651, 0.6300629, 0.602802, 0.5760604, 0.55837345, 0.53822356, -0.46732038], [0.9828675, 0.97948223, 0.977697, 0.97436255, 0.9708976, 0.96551025, 0.9622785, 0.9601396, 0.95474344, 0.9502484, 0.9432431, 0.9376823, 0.9251504, 0.9188809, 0.9142719, 0.9051502, 0.8950961, 0.8841326, 0.87730145, 0.87046957, 0.853398, 0.8470144, 0.82933825, 0.81342155, 0.7928765, 0.77553016, 0.7644428, 0.74232537, 0.7243909, 0.7075102, 0.691905, 0.6797922, 0.6654936, 0.65225226, 0.5768539, 0.54806876, 0.53242993, 0.50897723, 0.48672494, -0.47931692], [0.9828675, 0.9816459, 0.97768205, 0.97583395, 0.97428256, 0.9716968, 0.97026294, 0.96801424, 0.96438605, 0.96129566, 0.9586465, 0.9545027, 0.9483294, 0.93837523, 0.929607, 0.9160323, 0.91120225, 0.8991915, 0.87491584, 0.86381096, 0.8558707, 0.8411967, 0.8338564, 0.82203186, 0.8163084, 0.7868977, 0.7714323, 0.7580583, 0.7482155, 0.7130368, 0.70288575, 0.6872864, 0.64415556, 0.6324775, 0.59994906, 0.5773175, 0.5510462, 0.523443, -0.48080584], [0.9828675, 0.9797492, 0.97852, 0.977428, 0.97450745, 0.9721023, 0.9700144, 0.9677023, 0.9654528, 0.9626373, 0.96003145, 0.95792985, 0.9515645, 0.94882816, 0.9398792, 0.93555063, 0.9294391, 0.9274046, 0.9179806, 0.91384757, 0.90774554, 0.8998389, 0.8929954, 0.8803709, 0.86765605, 0.8364669, 0.8091662, 0.791466, 0.77572346, 0.75746804, 0.7446108, 0.7309072, 0.70928144, 0.686831, 0.66581005, 0.6097157, 0.5732175, 0.55134946, 0.5446143, 0.5195133, 0.48567152, -0.4746172]]}
    # graph_history(hist['picture_0'])
    iteration_hist(hist['picture_0'])
