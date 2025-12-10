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


def test_partition(df, model, column_name, threshold=None):
    # unequal partition, that is, if we it is the bias model we want to see the effect
    df_grater, df_smaller = partition(df, column_name, threshold)
    X_grater, y_grater = split_data(df_grater)
    X_smaller, y_smaller = split_data(df_smaller)

    onnx_inputs = {
        col: df[col].to_numpy(dtype=np.float32).reshape(-1, 1)
        for col in X_grater.columns
    }

    print(onnx_inputs.keys())

    y_pred_grater = model.run(None,  onnx_inputs)[0]
    y_pred_smaller = model.run(None,  onnx_inputs)[0]

    # count how many are positive in each partition
    greater_positive = sum(y_pred_grater)
    smaller_positive = sum(y_pred_smaller)

    #print(greater_positive, smaller_positive)

    return greater_positive, smaller_positive


def test_model(model, columns, df):
    model1_results = {}
    for col in columns:
        greater_positive, smaller_positive = test_partition(df, model, col, None)
        model1_results[col] = (greater_positive, smaller_positive)
    # check the average difference
    diffs = [abs(g - s) for g, s in model1_results.values()]

    return abs(np.mean(diffs))


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

    model = rt.InferenceSession('../team1/model_1.onnx')
    diff1 = test_model(model, cols_names, test_df)
    print('Model 1 average absolute difference:', diff1)

    model = rt.InferenceSession('../team1/model_2.onnx')
    diff2 = test_model(model, cols_names, test_df)
    print('Model 2 average absolute difference:', diff2)


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




