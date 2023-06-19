from pandas import DataFrame
from sklearn.preprocessing import StandardScaler


def center_data(df: DataFrame, axis=0):
    """
    Returns the same DataFrame but with centered data.
    :param df: DataFrame to be centered
    :param axis: axis to be used
    :return: pandas DataFrame
    """
    # centering the data
    return df - df.mean(axis=axis)


def normalize_data(df: DataFrame, axis=0):
    """
    Returns the same DataFrame but with normalized rows.
    :param df: DataFrame to be normalized
    :param axis: axis to be used
    :return: pandas DataFrame
    """
    # Normalize objects (rows) using the maximum absolute scaling
    # return (df - df.mean(axis=0)) / df.std(axis=0)
    # Normalize the data in pandas using Z-Score
    return df / df.std(axis=axis)


def standardize_data(df: DataFrame, axis=0):
    """
    Returns the same DataFrame standardized using Z-Score
    :param axis: axis to be used
    :param df: DataFrame to be standardized
    :return: pandas DataFrame
    """
    # Standardize the data in pandas using Z-Score
    return (df - df.mean(axis=axis)) / df.std(axis=axis)


def drop_insignificant_data(df: DataFrame, threshold=0):
    """
    Returns the same DataFrame without columns (genes) which have variance
    smaller than some threshold.
    :param threshold: the minimum variance of an attribute to be kept in df
    :param df: DataFrame for which zero-variance columns are to be removed
    :return: pandas DataFrame
    """
    # Create boolean sequence, True if variance of the gene is zero
    variance_is_zero = df.var() >= threshold
    # Reduce the DataFrame to genes whose variance is greater zero
    df = df.iloc[:, variance_is_zero.values]
    return df


def preprocessing(df: DataFrame, threshold=0):
    """
    The preprocessing includes:
    - dropping of columns that contain None/NaN values
    - normalization of object values (rows)
    - dropping columns with variance less than min_variance value
    :param threshold: the minimum variance of an attribute to be kept in df
    :param df: pandas DataFrame to be pre-processed
    :return: preprocessed pandas DataFrame
    """
    # Drop NaN values
    df = df.dropna(how='any', axis=1)
    # Standardize the data
    df = standardize_data(df)
    # Drop insignificant
    df = drop_insignificant_data(df, threshold=threshold)
    return df


def preprocessing_sklearn(df: DataFrame, threshold=0):
    """
    The preprocessing includes:
    - dropping of columns that contain None/NaN values
    - normalization of object values (rows)
    - dropping columns with variance less than min_variance value
    :param threshold: the minimum variance of an attribute to be kept in df
    :param df: pandas DataFrame to be pre-processed
    :return: preprocessed pandas DataFrame
    """
    # Drop NaN values
    df = df.dropna(how='any', axis=1)
    # Standardize the data
    df = StandardScaler().fit_transform(df)
    # Drop insignificant
    df = drop_insignificant_data(df, threshold=threshold)
    return df
