import pandas as pd
from typing import List, Union, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder



def load_spotify_dataset(path=None):
    """
    Load the Spotify dataset from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not path:
        raise ValueError("Path to file not found.")

    data = pd.read_csv(path)

    return data


def load_X_y(path: Optional[str] = None, attribute_list: Union[str, List[str]] = None, sample_size: Optional[int] = None):
    """
    Load the dataset and preprocess it for clustering. It drops all eventually existing duplicates.
    There are no NaN existing. Because 'Children's Music' is existing twice, we delete the smaller part with 5403 Records.
    The dataset is furthermore sampled to generate a smaller dataset for testing the clustering.
    If we take more than 100 samples we need to delete the genre 'A Capella' from our dataset.
    Afterwards we can apply a custom attribute selection or a prepared selection with all numeric features.
    The feature matrix X is then standardized with the MinMaxScaler, because many of the feature values already are in between 0 and 1 (optionally StandardScaler).
    The labeled data y ('genre') is encoded to numeric values.

    Args:
        path (str): Path to the CSV file.
        attribute_list (Union[str, List[str]]): List of attributes to include in X.
        sample_size (int): Number of samples per genre.

    Returns:
        np.ndarray: Preprocessed feature matrix X.
        np.ndarray: Encoded labeled vector y.
        DataFrame: Sampled subset of the Spotify dataset
    """

    data = load_spotify_dataset(path)  

    #  Prepare Dataset
    data = data.drop_duplicates()
    data = data[data['genre'] != "Children's Music"]

    #  Sample Dataset
    if sample_size == None:
        data = data.groupby('genre').sample(n=100,random_state=42)
    elif sample_size > 100:
        data = data[data['genre'] != "A Capella"]  # Exclude genre A Capella (119 records)
        data = data.groupby('genre').sample(n=sample_size,random_state=42)
    else:
        data = data.groupby('genre').sample(n=sample_size,random_state=42)

    #  Attribute Selection
    if attribute_list is None:
        selected_columns = [col for col in data.columns if col not in ['genre','artist_name','track_name', 'track_id','key','mode','time_signature','popularity']]
    elif isinstance(attribute_list, list):
        selected_columns = attribute_list
    else:
        raise ValueError("Invalid attribute_list value.")

    print(selected_columns)

    X = data[selected_columns]

    # Z-score Standardisation of X
    #zscore_scaler = StandardScaler()
    #X = zscore_scaler.fit_transform(X)

    # MinMaxScaler of X
    minmaxscaler = MinMaxScaler()
    X = minmaxscaler.fit_transform(X)

    # Encode label y
    label_encoder = LabelEncoder()
    data['genre_numeric'] = label_encoder.fit_transform(data['genre'])
    y = data['genre_numeric'].values.reshape(-1, 1)
    
    return X, y, data

#load_spotify_dataset(path='./SpotifyFeatures.csv')
#load_X_y(path='./SpotifyFeatures.csv')