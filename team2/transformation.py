from random import randint
import numpy as np


def transform_borrow(df):
    # get all columns that start with 'adres_recentste_wijk
    address_columns = [col for col in df.columns if col.startswith('adres_recentste_wijk')]
    n_borrows = len(address_columns)
    df['current_borrow_index'] = df[address_columns].values.argmax(axis=1)
    df['new_borrow_index'] = df['current_borrow_index'].apply(lambda x: (x + randint(1, n_borrows - 1)) % n_borrows)
    # zero out all address columns
    zeros = np.zeros((df.shape[0], n_borrows), dtype=int)
    zeros[np.arange(df.shape[0]), df['new_borrow_index']] = 1
    df[address_columns] = zeros

    df.drop(columns=['current_borrow_index', 'new_borrow_index'], inplace=True)
    return df
