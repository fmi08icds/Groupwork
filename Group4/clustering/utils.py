import numpy as np
from numpy.typing import NDArray


def distance_to_reference(points: NDArray, reference: NDArray) -> NDArray:
    """Compute euclidean distance to a reference point for each sample in a points array

    Parameters:
        points: Points array of shape (n_samples, n_features)
        reference: Reference point as an array of shape (n_features,)

    Returns: Array of shape (n_sampes,) with the euclidean distances to the reference point for each sample
    """
    # Check shape of parameters
    if len(points.shape) != 2 or len(reference.shape) != 1 or points.shape[1] != reference.shape[0]:
        raise ValueError(
            "Parameters have invalid or incompatible shapes! Expected (n_samples, n_features) (n_features,) but found "
            f"{points.shape} {reference.shape}"
        )
    # Compute euclidean distance, i. e., the square root of the sum of squared differences
    return np.sqrt(np.sum((points - reference.reshape(1, -1)) ** 2, axis=1))
