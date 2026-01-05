

# Model Training and Testing Instructions

This repository contains work from **two teams (Team 1 and Team 2)**.
Each team trains its own models and provides tests that can be run on **either team’s models**.

---

## Step 1: Train the Models

### Team 1 – Train Models

1. Navigate to the `team1` folder

2. Open and run the notebook: fraud_detection.ipynb

3. Run **all cells** in the notebook.

---

### Team 2 – Train Models

1. Navigate to the `team2` folder


2. Make sure that a folder named `data` exists **in the same directory** as `main.py`.

   * The `data` folder **can be empty or non-empty**
   * Its contents do not matter for training
   * Only its existence is required
   
3. Run the training script: main.py

---

## Step 2: Run Team 1 Tests

Team 1 tests are notebook-based and can be used to test **models from both teams**.

1. Navigate to the `team1` folder and notebook test_fraud_model.ipynb

2. Scroll to the **last cell** of the notebook.

3. Change the `MODEL_PATH` variable depending on which models you want to test.

### To test Team 1 models

Set `MODEL_PATH` to one of the following:

* model_1.onnx
* model_2.onnx

### To test Team 2 models

Set `MODEL_PATH` to one of the following:

* ../team2/models/model1.onnx
* ../team2/models/model2.onnx

4. Run the last cell to execute the tests.

---

## Step 3: Run Team 2 Tests

Team 2 tests are script-based and can also be run on **models from both teams**.

1. Navigate to the `team2` and open file tests.py 


2In the `__main__` section, locate the model loading lines.

### To test Team 1 models

Use:

* model_1 = rt.InferenceSession('../team1/model_1.onnx')
* model_2 = rt.InferenceSession('../team1/model_2.onnx')

### To test Team 2 models

Change the paths to:

* model_1 = rt.InferenceSession('models/model1.onnx')
* model_2 = rt.InferenceSession('models/model2.onnx')

Lines pointing to `../team1/...` correspond to **Team 1 models**.
Lines pointing to `models/...` correspond to **Team 2 models**.

3. Save the file.
4. Run the tests:  tests.py
