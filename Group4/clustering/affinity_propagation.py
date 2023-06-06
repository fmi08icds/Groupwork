import numpy as np
from numpy.typing import NDArray
from typing import Optional

from utils import squared_euclidean_distance


def compute_responsibility(similarities_mat: NDArray,
                           responsibilities_mat: NDArray,
                           availabilities_mat: NDArray,
                           damping: float = 0.5) -> NDArray:
    """ Computes values for the responsibility messages for the next iteration.
        Parameters:
            similarities_mat:       Array of similarity values; Shape `(n, n)`
            responsibilities_mat:   Array of responsibilities from previous iteration; Shape `(n, n)`
            availabilities_mat:     Array of availabilities from previous iteration; Shape `(n, n)`
            damping:                Damping factor to avoid numerical oscillations between iterations.
                                    0: Only new responsibility values will be used
                                    1: Only old responsibility values will be used (no update)
                                    Recommended range [0.5, 1.0)
        Returns:
            Array of shape `(n, n)` containing the new responsibility values.
    """

    # 1. Add availability and similarity for all exemplars (excluding self-reference [the diagonal])
    sum_mat = availabilities_mat + similarities_mat
    np.fill_diagonal(sum_mat, -np.inf)

    # 2. Select the column with the maximum value (best suited exemplar) for each row (point to be assigned an exemplar)
    row_indices = np.arange(sum_mat.shape[0])
    max_indices = np.argmax(sum_mat, axis=1)
    row_max = sum_mat[row_indices, max_indices]

    # 3. Update row_max to -np.inf to get second-best exemplar (That competes with the best exemplar)
    sum_mat[row_indices, max_indices] = -np.inf
    secondary_row_max = sum_mat[row_indices, np.argmax(sum_mat, axis=1)]

    # 4. Create a matrix with the maximum sum of availability and responsibility;
    # Set values for max_indices to secondary_row_max (Will be subtracted from similarity values)
    max_sum = np.zeros_like(similarities_mat) + row_max.reshape(-1, 1)
    max_sum[row_indices, max_indices] = secondary_row_max

    # 5. Calculate new responsibility values and return update after damping
    new_responsibility_mat = similarities_mat - max_sum
    return (1 - damping) * new_responsibility_mat + damping * responsibilities_mat


def compute_availability(responsibilities_mat: NDArray,
                         availabilities_mat: NDArray,
                         damping: float = 0.5) -> NDArray:
    """ Computes values for the availability messages for the next iteration.
        Parameters:
            responsibilities_mat:   Array of responsibilities from previous iteration; Shape `(n, n)`
            availabilities_mat:     Array of availabilities from previous iteration; Shape `(n, n)`
            damping:                Damping factor to avoid numerical oscillations between iterations.
                                    0: Only new responsibility values will be used
                                    1: Only old responsibility values will be used (no update)
                                    Recommended range [0.5, 1.0)
        Returns:
            Array of shape `(n, n)` containing the new availability values.
    """
    # 1. Create a temporary matrix to find the sum of responsibilities for each exemplar to points other than itself
    # - Only consider non-negative values in the sum    -> Set negative values to 0
    # - Only consider responsibilities to other points  -> Set diagonal values to 0
    tmp = responsibilities_mat.copy()
    tmp = np.where(tmp < 0, 0, tmp)
    np.fill_diagonal(tmp, 0)
    responsibilities_sum = np.sum(tmp, axis=0)  # Shape (n, )

    # 2. Get self responsibilities; Shape (n, )
    self_responsibilities = np.diag(responsibilities_mat).copy()

    # 3. Calculate new availability values
    # - self_responsibilities + responsibilities_sum is calculated on exemplar level -> the same for each column
    # - The availability of point e to point i is given by the sum of non-negative responsibilities e has to points j
    #   other than e or i. Hence, tmp needs to be subtracted [contains non-negative r(i, e)]
    availabilities_mat_new = np.minimum(0, self_responsibilities + responsibilities_sum - tmp)

    # 4. Set self availabilities (Sum of non-negative responsibilities to other points) and return update after damping
    np.fill_diagonal(availabilities_mat_new, np.sum(tmp, axis=0))
    return (1 - damping) * availabilities_mat_new + damping * availabilities_mat


def compute_similarity(X: np.ndarray, precomputed: Optional[NDArray] = None) -> NDArray:
    """Compute similarity values between points in array X. Default are negative Euclidean distances
    Parameters:
        X:              Array with n rows (i.e. points)
        precomputed:    Optional array of shape (n, n) to provide precomputed similarities

    Returns:
        Array of shape `(n, n)` containing similarity values.
    """
    if precomputed is not None:
        n = X.shape[0]
        assert precomputed.shape == (n, n), \
            print(f'The precomputed similarities need to be of shape ({n}, {n})')
        return precomputed

    return squared_euclidean_distance(X)


def set_preferences(similarities_mat: np.ndarray, preferences: Optional[NDArray] = None) -> NDArray:
    """Updates the similarity_matrix with preferences values
    Parameters:
        similarities_mat:  Array of shape (n, n) containing similarities between n points
        preferences:        Optional array of shape (n, ); If None, the median of all similarities is used

    Returns:
        Array of updated similarity values; Shape (n, n)
    """
    # Calculate the median similarity or confirm correct input of preferences vector
    if preferences is None:
        preferences = np.median(similarities_mat)
    else:
        n = similarities_mat.shape[0]
        assert preferences.shape == (n,), print(f'Preferences must be None or an array of shape ({n},)')

    # Update self similarity values and return updated matrix
    np.fill_diagonal(similarities_mat, preferences)
    return similarities_mat