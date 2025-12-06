import sklearn
import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

import onnxruntime as rt
import onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx
from skl2onnx import convert_sklearn
from sklearn.metrics import accuracy_score


def bad_model(X, y):
    # 318 columns -3 for answers, 315 decision variables
    model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=30)
    model.fit(X, y)
    return model


def test_model(X, y, model):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"Model accuracy: {accuracy}")
    return accuracy


def check_feature_importance(model, feature_names):
    importances = model.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    print("Feature importances:")
    importances = zip(feature_names, importances)
    importances = sorted(importances, key=lambda x: x[1], reverse=True)
    for feature, importance in importances:
        if importance > 0:
            print(f"{feature}: {importance}")


if __name__ == "__main__":
    # ,Ja,Nee,checked
    """
    X = pd.read_csv('data/local_train.csv').drop(columns=["Ja", "Nee", "checked"]).astype(np.float32)
    y = pd.read_csv('data/local_train.csv')["checked"]
    model = bad_model(X, y)
    with open('models/bad_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    onnx_model = convert_sklearn(
        model,
        initial_types=[('X', FloatTensorType([None, len(X.columns)]))],
        target_opset=12
    )


    X_test = pd.read_csv('data/local_test.csv').drop(columns=["Ja", "Nee", "checked"])
    y_test = pd.read_csv('data/local_test.csv')["checked"]
    test_model(X_test, y_test, model)

    feature_names = pd.read_csv('data/local_train.csv').drop(columns=["Ja", "Nee", "checked"]).columns

    check_feature_importance(model, feature_names)

    onnx.save_model(onnx_model, 'models/bad_model.onnx')"""

    X_test = pd.read_csv('data/local_test.csv').drop(columns=["Ja", "Nee", "checked"])
    y_test = pd.read_csv('data/local_test.csv')["checked"]

    loaded_model = rt.InferenceSession('models/bad_model.onnx')

    y_pred_onnx = loaded_model.run(None, {'X': X_test.astype(np.float32).to_numpy()})[0]
    print(sum(y_pred_onnx))
    print(max(y_pred_onnx))
    accuracy_onnx = accuracy_score(y_test, y_pred_onnx)
    print(f"ONNX Model accuracy: {accuracy_onnx}")
