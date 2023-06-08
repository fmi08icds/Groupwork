import numpy as np
from numpy.typing import NDArray
from numpy.random import RandomState
from typing import Optional, Tuple

from clustering.utils import squared_euclidean_distance

# TODO: For a lower number of centers, my implementation returns more centers
#  - Check sklearn https://github.com/scikit-learn/scikit-learn/blob/364c77e04/sklearn/cluster/_affinity_propagation.py#L181

def affinity_propagation(X: NDArray, damping: float = 0.5, max_iter: int = 200, convergence_iter: int = 15,
                         affinity: Optional[NDArray] = None, preferences: Optional[NDArray] = None,
                         verbose: bool = False, random_state: Optional[int] = None) \
        -> Tuple[NDArray, NDArray, NDArray]:
    """Perform clustering on the point array `X` using the Affinity Propagation [1] algorithm.
    Parameters:
        X:                      Two-dimensional array of shape `(n_samples, n_features)`
        damping:                Damping factor to avoid numerical oscillations between iterations.
                                0: Only new responsibility values will be used
                                1: Only old responsibility values will be used (no update)
                                Recommended interval is [0.5, 1.0)
        max_iter:               Upper limit for number iterations for message passing
        convergence_iter:       Upper limit for number of iterations without change in labels
        affinity:               Optional array of shape (n, n) to provide precomputed similarities
        preferences:            Optional array of shape (n, ); If None, the median of all similarities is used
        verbose:                True: Print number of iterations
        random_state:           Integer to initialize an instance of numpy.random.RandomState
                                (Used to remove degeneracies in the similarity matrix)

    Returns:
        cluster_centers_indices:    Array of indices of the points used as cluster centers (exemplars)
        cluster_centers             Array of coordinates of the points used as cluster centers (exemplars)
        final_labels                Array of shape `(n, )` with each point being assigned a cluster center (exemplar)
    References:
        [1] B. J. Frey and D. Dueck. Clustering by Passing Messages Between Data Points.
            In: Science 315, pp. 972-976 (2007).DOI:10.1126/science.1136800
    """
    assert 0 <= damping <= 1, print('The damping factor must be in [0, 1]; The recommended interval is [0.5, 1.0)')
    random_state = RandomState(seed=random_state)

    # 1. Preparation: Calculate similarities, set optional preferences, and initialize message values
    similarities_mat, responsibilities_mat, availabilities_mat = prepare_matrices(X=X,
                                                                                  random_state=random_state,
                                                                                  affinity=affinity,
                                                                                  preferences=preferences)

    # 2. Loop for max iterations or until no change was recorded for convergence_iter
    i, iter_no_change = 0, 0
    for i in range(max_iter):
        # Calculate the scores to find current labels
        score = responsibilities_mat + availabilities_mat
        labels = np.argmax(score, axis=1)

        # Update message values
        responsibilities_mat = update_responsibility(similarities_mat, responsibilities_mat, availabilities_mat,
                                                     damping=damping)
        availabilities_mat = update_availability(responsibilities_mat, availabilities_mat, damping=damping)

        # Update labels and iterations without a change
        new_score = responsibilities_mat + availabilities_mat
        new_labels = np.argmax(new_score, axis=1)

        if np.all(new_labels == labels):
            iter_no_change += 1
        else:
            iter_no_change = 0

        # If the convergence iterations are reached, break the loop
        if iter_no_change >= convergence_iter:
            break

    if verbose:
        msg = f"Converged after {i} iterations." if i < (max_iter - 1) else \
            f"Reached {max_iter} iterations without convergence."
        print(msg)

    # 3. Finalize
    # 3a. Assign points to the final cluster_centers (exemplars)
    final_scores = responsibilities_mat + availabilities_mat
    final_labels = np.argmax(final_scores, axis=1)
    cluster_centers_indices = np.unique(final_labels)
    cluster_centers = X[cluster_centers_indices]

    # 3b. Replace cluster_center_indices by integers from 0 to num_centers - 1; Update labels
    mapping = np.arange(0, max(final_labels) + 1)
    mapping[list(cluster_centers_indices)] = list(range(len(cluster_centers_indices)))
    final_labels = mapping[final_labels]

    return cluster_centers_indices, cluster_centers, final_labels

def update_responsibility(similarities_mat: NDArray,
                          responsibilities_mat: NDArray,
                          availabilities_mat: NDArray,
                          damping: float = 0.5) -> NDArray:
    """Compute values for the responsibility messages for the next iteration.
    Parameters:
        similarities_mat:       Array of similarity values; Shape `(n, n)`
        responsibilities_mat:   Array of responsibilities from previous iteration; Shape `(n, n)`
        availabilities_mat:     Array of availabilities from previous iteration; Shape `(n, n)`
        damping:                Damping factor to avoid numerical oscillations between iterations.
                                0: Only new responsibility values will be used
                                1: Only old responsibility values will be used (no update)
                                Recommended interval is [0.5, 1.0)
    Returns:
        Array of shape `(n, n)` containing the new responsibility values.
    """
    row_index = np.arange(availabilities_mat.shape[0])

    # 1. Compute competing responsibilities
    sum_mat = availabilities_mat + similarities_mat

    # 2. Select the column with the maximum value (best suited exemplar) for each row (point to be assigned an exemplar)
    max_index = np.argmax(sum_mat, axis=1)
    row_max = sum_mat[row_index, max_index]

    # 3. Update row_max to -np.inf to get second-best exemplar (That competes with the best exemplar)
    sum_mat[row_index, max_index] = -np.inf
    competing_resp = np.max(sum_mat, axis=1)

    # 4. Calculate the new responsibility values
    responsibilities_mat_new = similarities_mat - row_max[:, None]
    responsibilities_mat_new[row_index, max_index] = similarities_mat[row_index, max_index] - competing_resp

    # 5. Return result after damping
    return (1 - damping) * responsibilities_mat_new + damping * responsibilities_mat

def update_availability(responsibilities_mat: NDArray,
                        availabilities_mat: NDArray,
                        damping: float = 0.5) -> NDArray:
    """Compute values for the availability messages for the next iteration.
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


def compute_similarity(X: np.ndarray, affinity: Optional[NDArray] = None) -> NDArray:
    """Compute similarity values between points in array X. Default are negative Euclidean distances
    Parameters:
        X:           Two-dimensional array of shape `(n, m)` (n = number of points; m = number of features)
        affinity:    Optional array of shape (n, n) to provide precomputed similarities

    Returns:
        Array of shape `(n, n)` containing similarity values.
    """
    if affinity is not None:
        n = X.shape[0]
        assert affinity.shape == (n, n), \
            print(f'The provided affinity matrix need to be of shape ({n}, {n})')
        return affinity

    return squared_euclidean_distance(X)


def set_preferences(similarities_mat: np.ndarray, preferences: Optional[NDArray] = None) -> NDArray:
    """Update the similarity_matrix with preferences values
    Parameters:
        similarities_mat:   Array of shape (n, n) containing similarities between n points
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

def prepare_matrices(X: NDArray, random_state: RandomState, affinity: Optional[NDArray] = None,
                     preferences: Optional[NDArray] = None) -> Tuple[NDArray, NDArray, NDArray]:
    """Prepare the matrices for similarity, responsibility, and availability
    Parameters:
        X:              Two-dimensional array of shape `(n, m)` (n = number of points; m = number of features)
        random_state:   Instance of numpy.random.RandomState (Used to remove degeneracies in the similarity matrix)
        affinity:       Optional array of shape (n, n) to provide precomputed similarities
        preferences:    Optional array of shape (n, ); If None, the median of all similarities is used

    Returns:
                Three arrays of shape (n, n): Similarities, Responsibilities, Availabilities
    """
    S = compute_similarity(X, affinity)
    S = set_preferences(S, preferences=preferences)

    # Adapt similarity_mat to remove any degeneracies that might cause oscillation
    S += (np.finfo(S.dtype).eps * S + np.finfo(S.dtype).tiny * 100) \
         * random_state.standard_normal(size=(S.shape[0], S.shape[0]))

    return S, np.zeros_like(S), np.zeros_like(S)