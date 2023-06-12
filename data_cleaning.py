from pandas import DataFrame


def clean_data(df: DataFrame):
    """
    Returns the same DataFrame without columns (genes) which have variance zero
    :param df: DataFrame for which zero-variance columns are to be removed
    :return: pandas DataFrame
    """
    # Create boolean sequence, True if variance of the gene is zero
    variance_is_zero = df.var() != 0
    # Reduce the DataFrame to genes whose variance is greater zero
    df = df.iloc[:, variance_is_zero.values]
    return df
