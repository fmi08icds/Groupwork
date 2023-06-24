import numpy as np

from collections import deque

from typing import Tuple
from numpy.typing import NDArray

from clustering.utils import distance_to_reference


def dbscan(X: NDArray, epsilon: float = 0.5, min_points: int = 5) -> Tuple[NDArray, NDArray]:
    """Perform clustering on the point array `X` using the Density-Based Spatial Clustering for Applications with
    Noise (DBSCAN) [1] algorithm.

    Parameters:
        X: Two-dimensional array of shape `(n_samples, n_features)`
        epsilon: Maximum Euclidean distance between two points to consider one in the neighborhood of the other
        min_points: Defines how many points (including itself) must be in the neighborhood to consider the point a
            core point. A common heuristic is to pick `2 * n_features` [2]

    Returns:
        core_sample_indices:    Array of shape `(n_core_samples,)` with indices of core samples.
                                Difference to sklearn implementation: Only previously not visited points are recorded.
                                Hence, this output is equal to the "actual" cluster centers
        labels                  Array of shape `(n_samples,)` with the cluster index for each sample.
                                If the index is `-1`, the point is considered an outlier.


    References:
        [1] M. Ester, H. Kriegel, J. Sander and X. Xu. A density-based algorithm for discovering clusters in large
            spatial databases with noise. In: Proceedings of the Second International Conference on Knowledge Discovery
            and Data Mining, 1996, pp. 226-231.
        [2] J. Sander, M. Ester, H. Kriegel and X. Xu. Density-Based Clustering in Spatial Databases: The Algorithm
            GDBSCAN and Its Applications. In: Data Mining and Knowledge Discovery 2, 1998, pp. 169-194.
    """

    # Make sure that input array is two-dimensional
    if len(X.shape) != 2:
        raise ValueError(f"Parameter X must be of shape (n_samples, n_features) instead of {X.shape}")

    n_samples = X.shape[0]

    # Mark all points as outliers until they are assigned a cluster index (starting with 0)
    labels = np.full(n_samples, fill_value=-1, dtype=np.int32)
    core_samples = np.full(n_samples, fill_value=-1, dtype=np.int32)
    cluster_index = 0

    # Keep track of points that have already been visited
    point_visited = np.full(n_samples, fill_value=False, dtype=bool)

    # Iterate over all samples
    point_indices = np.arange(n_samples)
    for i in point_indices:
        # Skip already visited points
        if point_visited[i]:
            continue

        # Mark point as visited
        point_visited[i] = True

        # Get point indices of directly density-reachable neighbors (including the current point itself)
        directly_reachable = point_indices[distance_to_reference(X, X[i]) <= epsilon]
        # If core point condition is not satisfied, the point remains noise for now
        if len(directly_reachable) < min_points:
            continue

        # Iteratively find reachable points to expand the cluster until all points have been processed
        reachable = deque(directly_reachable)
        while len(reachable) > 0:
            # Get "oldest" element from the deque
            j = reachable.popleft()

            # Assign reachable point to current cluster if it does not already belong to a cluster
            if labels[j] == -1:
                labels[j] = cluster_index

            # Ignore points that have already been visited
            if not point_visited[j]:
                point_visited[j] = True
                # Add new reachable points to the deque (expand the cluster)
                potentially_reachable = point_indices[distance_to_reference(X, X[j]) <= epsilon]
                if len(potentially_reachable) >= min_points:
                    reachable.extend(potentially_reachable)

        # Cluster is now complete, increment cluster index for next cluster
        core_samples[i] = cluster_index
        cluster_index += 1

    core_samples = np.asarray(core_samples > -1, dtype=np.uint8)
    core_sample_indices = np.where(core_samples)[0]

    return core_sample_indices, labels
