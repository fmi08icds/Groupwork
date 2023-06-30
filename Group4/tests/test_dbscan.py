import numpy as np
from sklearn.cluster import dbscan as sklearn_dbscan
from sklearn.datasets import make_blobs
from numpy.testing import assert_array_equal

from clustering.dbscan import dbscan

import pytest


@pytest.mark.parametrize("n_centers", [1, 2, 4, 8])
@pytest.mark.parametrize("n_features", [2, 3, 5, 10])
@pytest.mark.parametrize("epsilon", [0.5, 1, 2, 3])
def test_dbscan_compare_results(n_centers: int, n_features: int, epsilon: float):
    # Generate dataset with `n_centers` blobs (circular clusters)
    X, _ = make_blobs(n_samples=n_centers * 70, centers=n_centers, n_features=n_features)

    # Run own implementation and scikit-learn version with the same parameters and compare the results
    _, computed_labels = dbscan(X, epsilon=epsilon, min_points=2 * n_features)
    _, expected_labels = sklearn_dbscan(X, eps=epsilon, min_samples=2 * n_features)
    assert_array_equal(computed_labels, expected_labels)
