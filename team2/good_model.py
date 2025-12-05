import numpy as np
import pandas as pd
import sklearn
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import onnxruntime as rt
import onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx
from skl2onnx import convert_sklearn


"""
Since we dont have actual data who commited fraud, only who has been checked. 
To get proper training data it would be good to filter out who was checked due to biases and who poses a real risk.

How can we do that?
- define set o sensitive variables, (adres, women and so on)
- train bad model on everything 
- permutate vulnerable variables 
- make new training set 

Why is it better then just permutating sensitive variables?

If a decision to check was made based on the vulnerable variables changing them does not change the fact that
the true label is checked, this is a problem since it leads to some datapoints being labeled true even if they
 should not be...

whatever just permutate, smaller model  
"""

"""
bad columns:
1-23, addresses
58,59,65,66,67 - psycho and health
74-91 - comepetentie
92-153 - subjective
216, 217 - gender_women and age
218-252 - subjective
283-304 - personal relations
"""


class GoodModel:

    def __init__(self, bad_column_names, X, y):
        self.bad_column_names = bad_column_names
        self.good_cols = [i for i, col in enumerate(X.columns) if col not in self.bad_column_names]
        self.model = self.train(X, y)

    def train(self, X, y):
        good_cols = self.good_cols
        preprocess = ColumnTransformer([
            ("keep", "passthrough", good_cols),
        ],
        remainder="drop"
        )
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        pipeline = Pipeline(steps=[
            ('preprocess', preprocess),
            ('classifier', model)
        ])
        pipeline.fit(X, y)
        return pipeline

    def predict(self, X):
        X_good = X.drop(columns=self.bad_column_names)
        return self.model.predict(X_good)

    def test(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        print(f"Good Model accuracy: {accuracy}")
        return accuracy


if __name__ == '__main__':
    info_df = pd.read_csv('../data/data_description.csv', encoding='windows-1252', index_col=0)
    cols_ids = list(range(1, 24)) + [58, 59, 65, 66, 67, 216, 217] + list(range(74, 92)) + list(
        range(92, 154)) + list(range(218, 253)) + list(range(283, 305))
    print(info_df)
    cols_names = info_df.loc[cols_ids]['Feature (nl)'].values.tolist()
    cols_names = [n for n in cols_names if n not in ['competentie_overtuigen_en_beïnvloeden', 'contacten_onderwerp_boolean_financiële_situatie', 'contacten_onderwerp_financiële_situatie'] ]

    X_train = pd.read_csv('data/local_train.csv').drop(columns=["Ja", "Nee", "checked"]).astype(np.float32)
    y_train = pd.read_csv('data/local_train.csv')["checked"]

    m = GoodModel(cols_names, X_train, y_train)
    t = m.model.named_steps['preprocess'].transform(X_train)
    print(t.shape)
    print(X_train.shape)
    df = pd.read_csv('data/local_train.csv').drop(columns=["Ja", "Nee", "checked"])
    with open('models/good_model.pkl', 'wb') as f:
        pickle.dump(m, f)

    onnx_model = convert_sklearn(
        m.model,
        initial_types=[('X', FloatTensorType([None, len(df.columns)]))],
        target_opset=12
    )

    onnx.save_model(onnx_model, 'models/good_model.onnx')

    loaded_model = rt.InferenceSession('models/good_model.onnx')

    X_test = pd.read_csv('data/local_test.csv').drop(columns=["Ja", "Nee", "checked"]).astype(np.float32)
    y_test = pd.read_csv('data/local_test.csv')["checked"]

    print(m.test(X_test, y_test))

    y_pred_onnx = loaded_model.run(None, {'X': X_test.astype(np.float32).to_numpy()})[0]
    accuracy_onnx = accuracy_score(y_test, y_pred_onnx)

    print(f"ONNX Model accuracy: {accuracy_onnx}")


