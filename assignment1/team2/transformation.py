from random import randint
import numpy as np


def range_transformation(df, prefix='adres_recentste_wijk'):
    # get all columns that start with 'adres_recentste_wijk
    range_columns = [col for col in df.columns if col.startswith(prefix)]
    n_borrows = len(range_columns)
    df['current_range_index'] = df[range_columns].values.argmax(axis=1)
    df['new_range_index'] = df['current_range_index'].apply(lambda x: (x + randint(1, n_borrows - 1)) % n_borrows)
    # zero out all address columns
    zeros = np.zeros((df.shape[0], n_borrows), dtype=int)
    zeros[np.arange(df.shape[0]), df['new_range_index']] = 1
    df[range_columns] = zeros

    df.drop(columns=['current_range_index', 'new_range_index'], inplace=True)
    return df


def point_transformation(df, column_name):
    min_value = df[column_name].min()
    max_value = df[column_name].max()

    # assume min-max is the full range of the variable

    df[column_name] = np.random.randint(min_value, max_value + 1, size=df.shape[0])

    return df