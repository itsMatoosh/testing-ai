import sklearn
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def bad_model(X, y):
    # 318 columns -3 for answers, 315 decision variables
    model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=30)
    model.fit(X, y)
    return model


def test_model(X, y, model):
    predictions = model.predict(X)
    accuracy = sklearn.metrics.accuracy_score(y, predictions)
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
    """
    # ,Ja,Nee,checked
    X = pd.read_csv('data/local_train.csv').drop(columns=["Ja", "Nee", "checked"])
    y = pd.read_csv('data/local_train.csv')["checked"]
    model = bad_model(X, y)
    with open('models/bad_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    X_test = pd.read_csv('data/local_test.csv').drop(columns=["Ja", "Nee", "checked"])
    y_test = pd.read_csv('data/local_test.csv')["checked"]
    test_model(X_test, y_test, model)
    """
    feature_names = pd.read_csv('data/local_train.csv').drop(columns=["Ja", "Nee", "checked"]).columns
    with open('models/bad_model.pkl', 'rb') as f:
        model = pickle.load(f)

    check_feature_importance(model, feature_names)
