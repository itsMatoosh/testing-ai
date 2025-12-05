import sklearn
import pickle
import pandas as pd
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


if __name__ == "__main__":
    # ,Ja,Nee,checked
    X = pd.read_csv('data/local_train.csv').drop(columns=["Ja", "Nee", "checked"])
    y = pd.read_csv('data/local_train.csv')["checked"]
    model = bad_model(X, y)
    with open('models/bad_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    X_test = pd.read_csv('data/local_test.csv').drop(columns=["Ja", "Nee", "checked"])
    y_test = pd.read_csv('data/local_test.csv')["checked"]
    test_model(X_test, y_test, model)
