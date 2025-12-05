

def partition(data, column_name, threshold):
    """
    Partitions the DataFrame into two based on a threshold value for a specified column.

    Parameters:
    data (pd.DataFrame): The input DataFrame to be partitioned.
    column_name (str): The name of the column to apply the threshold on.
    threshold (float): The threshold value for partitioning.

    Returns:
    tuple: A tuple containing two DataFrames:
        - The first DataFrame contains rows where the column value is greater than the threshold.
        - The second DataFrame contains rows where the column value is less than or equal to the threshold.
    """
    greater_than_threshold = data[data[column_name] > threshold]
    less_equal_threshold = data[data[column_name] <= threshold]

    return greater_than_threshold, less_equal_threshold


def unequal_partition(df):
    return partition(df, 'persoonlijke_eigenschappen_taaleis_voldaan', 0.5)


def equivalence_partition(df):
    return partition(df, 'persoonlijke_eigenschappen_hobbies_sport', 0.5)