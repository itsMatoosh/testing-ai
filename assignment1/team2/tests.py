import os
import pickle

import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import onnxruntime as rt
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from partition import *
from transformation import *


def predict_onnx(session, X_input):
    """
    Generates predictions using an ONNX session.
    Assumes all required fields exist in X_input.
    """
    input_meta = session.get_inputs()
    inputs = {}

    for meta in input_meta:
        name = meta.name
        # If the model expects specific features by name, we extract them.
        if name in X_input.columns:
            inputs[name] = X_input[name].values.astype(np.float32).reshape(-1, 1)
        elif len(input_meta) == 1:
            # Fallback for single matrix input
            inputs[name] = X_input.to_numpy(dtype=np.float32)

    output_names = [meta.name for meta in session.get_outputs()]
    try:
        preds = session.run(output_names, inputs)
        return preds[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


def split_data(df):
    X = df.drop(columns=["Ja", "Nee", "checked"]).astype(np.float32)
    y = df["checked"]

    return X, y


def test_partition(df, model, column_name, threshold=None):
    # unequal partition, that is, if we it is the bias model we want to see the effect
    df_grater, df_smaller = partition(df, column_name, threshold)
    X_grater, y_grater = split_data(df_grater)
    X_smaller, y_smaller = split_data(df_smaller)

    y_pred_grater = predict_onnx(model, X_grater)
    y_pred_smaller = predict_onnx(model, X_smaller)

    # count how many are positive in each partition
    greater_positive = sum(y_pred_grater)
    smaller_positive = sum(y_pred_smaller)

    # print(greater_positive, smaller_positive)

    return greater_positive, smaller_positive


def test_model(model, columns, df):
    model1_results = {}
    for col in columns:
        greater_positive, smaller_positive = test_partition(df, model, col, None)
        model1_results[col] = (greater_positive, smaller_positive)
    # check the average difference
    diffs = [abs(g - s) for g, s in model1_results.values()]

    return diffs, abs(np.mean(diffs))

def classical_ml_evaluation(model, df):
    """
    Classical supervised ML evaluation metrics.
    """
    X, y_true = split_data(df)

    y_pred = predict_onnx(model, X)

    # Handle ONNX output shape
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = (y_pred > 0.5).astype(int).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }


if __name__ == "__main__":
    # generate bad columns
    info_df = pd.read_csv('../data/data_description.csv', encoding='windows-1252', index_col=0)
    cols_ids = list(range(1, 24)) + [58, 59, 65, 66, 67, 216, 217] + list(range(74, 92)) + list(
        range(92, 154)) + list(range(218, 253)) + list(range(283, 305))
    print(info_df)
    cols_names = info_df.loc[cols_ids]['Feature (nl)'].values.tolist()
    cols_names = [n for n in cols_names if n not in ['competentie_overtuigen_en_beïnvloeden', 'contacten_onderwerp_boolean_financiële_situatie', 'contacten_onderwerp_financiële_situatie'] ]

    test_df = pd.read_csv('data/global_test.csv')

    """
    diffs = []
    for col in cols_names:
        greater, smaller = partition(test_df, col, None)
        greater_positive = sum(greater['checked'])
        smaller_positive = sum(smaller['checked'])
        diffs.append(abs(greater_positive - smaller_positive))
    print('Data average absolute difference:', np.mean(diffs))"""
    path_to_diffs = 'data/diffs.pkl'
    # path_to_diffs = 'data/diffs_local.pkl'
    if not os.path.exists(path_to_diffs):

        model = rt.InferenceSession('../team1/model_1.onnx')
        # model = rt.InferenceSession('models/good_model.onnx')
        diffs1, diff1 = test_model(model, cols_names, test_df)
        print('Model 1 average absolute difference:', diff1)
        # classical ML evaluation
        metrics_1 = classical_ml_evaluation(model, test_df)
        print("Model 1 classical ML metrics:", metrics_1)

        model = rt.InferenceSession('../team1/model_2.onnx')
        # model = rt.InferenceSession('models/bad_model.onnx')
        diffs2, diff2 = test_model(model, cols_names, test_df)
        print('Model 2 average absolute difference:', diff2)

        metrics_1 = classical_ml_evaluation(model, test_df)
        print("Model 1 classical ML metrics:", metrics_1)

        with open(path_to_diffs, 'wb') as f:
            pickle.dump((diffs1, diffs2), f)
    else:
        with open(path_to_diffs, 'rb') as f:
            diffs1, diffs2 = pickle.load(f)
        print('Model 1 average absolute difference:', np.mean(diffs1))
        print('Model 2 average absolute difference:', np.mean(diffs2))

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    axs[0].hist(diffs1, bins=20, color='blue', alpha=0.7)
    axs[0].set_title('Model 1 Absolute Differences')
    axs[0].set_ylabel('Frequency')
    axs[1].hist(diffs2, bins=20, color='green', alpha=0.7)
    axs[1].set_title('Model 2 Absolute Differences')
    axs[1].set_xlabel('Absolute Difference')
    axs[1].set_ylabel('Frequency')

    plt.show()
    plt.savefig('data/differences_histogram.png')




    """
    model = rt.InferenceSession('models/model1.onnx')
    print('model_loaded')
    test_df = pd.read_csv('data/global_test.csv')
    test_partition(test_df, model, 'persoonlijke_eigenschappen_taaleis_voldaan', 0.5)
    model = rt.InferenceSession('models/model2.onnx')
    print('model_loaded')
    test_df = pd.read_csv('data/global_test.csv')
    test_partition(test_df, model, 'persoonlijke_eigenschappen_taaleis_voldaan', 0.5)
    """




