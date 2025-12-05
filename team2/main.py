import os
import pandas as pd
import sklearn
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import onnxruntime as rt
import onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx
from skl2onnx import convert_sklearn

from good_model import GoodModel
from bad_model import bad_model, test_model


def generate_local():
    df_desc = pd.read_csv('../data/data_description.csv', encoding='windows-1252')
    df = pd.read_csv('../data/investigation_train_large_checked.csv', encoding='windows-1252')
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv('data/global_train.csv', index=False)
    test.to_csv('data/global_test.csv', index=False)
    local_train, local_test = train_test_split(train, test_size=0.2, random_state=42)
    local_train.to_csv('data/local_train.csv', index=False)
    local_test.to_csv('data/local_test.csv', index=False)


def train_models():
    info_df = pd.read_csv('../data/data_description.csv', encoding='windows-1252', index_col=0)
    train_df = pd.read_csv('data/global_train.csv')
    X_train = train_df.drop(columns=["Ja", "Nee", "checked"]).astype(np.float32)
    y_train = train_df["checked"]

    # train good model
    cols_ids = list(range(1, 24)) + [58, 59, 65, 66, 67, 216, 217] + list(range(74, 92)) + list(
        range(92, 154)) + list(range(218, 253)) + list(range(283, 305))

    cols_names = info_df.loc[cols_ids]['Feature (nl)'].values.tolist()
    cols_names = [n for n in cols_names if
                  n not in ['competentie_overtuigen_en_beïnvloeden', 'contacten_onderwerp_boolean_financiële_situatie',
                            'contacten_onderwerp_financiële_situatie']]

    gm = GoodModel(cols_names, X_train, y_train)

    # secondary pickle save
    with open('models/good_model.pkl', 'wb') as f:
        pickle.dump(gm, f)

    onnx_model = convert_sklearn(
        gm.model,
        initial_types=[('X', FloatTensorType([None, len(X_train.columns)]))],
        target_opset=12
    )

    onnx.save_model(onnx_model, 'models/good_model.onnx')

    # train bad model

    bm = bad_model(X_train, y_train)
    with open('models/bad_model.pkl', 'wb') as f:
        pickle.dump(bm, f)

    onnx_model = convert_sklearn(
        bm,
        initial_types=[('X', FloatTensorType([None, len(X_train.columns)]))],
        target_opset=12
    )
    onnx.save_model(onnx_model, 'models/bad_model.onnx')

    # sanity check
    test_df = pd.read_csv('data/global_test.csv')
    X_test = test_df.drop(columns=["Ja", "Nee", "checked"]).astype(np.float32)
    y_test = test_df["checked"]

    print('good model test:')
    gm.test(X_test, y_test)
    print('good model onnnx:')
    loaded_model = rt.InferenceSession('models/good_model.onnx')
    y_pred_onnx = loaded_model.run(None, {'X': X_test.astype(np.float32).to_numpy()})[0]
    print(accuracy_score(y_test, y_pred_onnx))

    print('bad model test:')
    test_model(X_test, y_test, bm)
    print('bad model onnnx:')
    loaded_model = rt.InferenceSession('models/bad_model.onnx')
    y_pred_onnx = loaded_model.run(None, {'X': X_test.astype(np.float32).to_numpy()})[0]
    print(accuracy_score(y_test, y_pred_onnx))


if __name__ == "__main__":
    train_models()