import os
import pandas as pd

from sklearn.model_selection import train_test_split


def main():
    df_desc = pd.read_csv('../data/data_description.csv', encoding='windows-1252')
    df = pd.read_csv('../data/investigation_train_large_checked.csv', encoding='windows-1252')
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv('data/global_train.csv', index=False)
    test.to_csv('data/global_test.csv', index=False)
    local_train, local_test = train_test_split(train, test_size=0.2, random_state=42)
    local_train.to_csv('data/local_train.csv', index=False)
    local_test.to_csv('data/local_test.csv', index=False)


if __name__ == "__main__":
    main()