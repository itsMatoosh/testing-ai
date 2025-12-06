import pandas as pd
import numpy as np


import onnxruntime as rt
from sklearn.metrics import accuracy_score

from partition import *
from transformation import *


def split_data(df):
    X = df.drop(columns=["Ja", "Nee", "checked"]).astype(np.float32)
    y = df["checked"]

    return X, y


def test_partition(df, model, column_name, threshold):
    # unequal partition, that is, if we it is the bias model we want to see the effect
    df_grater, df_smaller = partition(df, column_name, threshold)
    X_grater, y_grater = split_data(df_grater)
    X_smaller, y_smaller = split_data(df_smaller)

    y_pred_grater = model.run(None, {'X': X_grater.astype(np.float32).to_numpy()})[0]
    y_pred_smaller = model.run(None, {'X': X_smaller.astype(np.float32).to_numpy()})[0]

    # count how many are positive in each partition
    print(y_pred_grater)
    greater_positive = sum(y_pred_grater)
    smaller_positive = sum(y_pred_smaller)

    print(greater_positive, smaller_positive)


if __name__ == "__main__":
    model = rt.InferenceSession('models/model_1_t1.onnx')
    print('model_loaded')
    test_df = pd.read_csv('data/global_test.csv')
    test_partition(test_df, model, 'persoonlijke_eigenschappen_taaleis_voldaan', 0.5)



