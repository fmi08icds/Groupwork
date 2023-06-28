import numpy as np
from sklearn.cluster import Birch as sklearn_birch
from sklearn.datasets import make_blobs
from numpy.testing import assert_array_equal

from clustering.birch import Birch

import pytest


@pytest.mark.parametrize("n_centers", [2, 4, 8])
@pytest.mark.parametrize("n_features", [2, 5, 10])
@pytest.mark.parametrize("branching_factor", [3, 5, 10])
@pytest.mark.parametrize("threshold", [.2, .5, 1])
def test_birch_compare_results(n_centers: int, n_features: int, branching_factor: int, threshold: float):
    # Generate dataset with `n_centers` blobs (circular clusters)
    X, _ = make_blobs(n_samples=n_centers * 50, centers=n_centers, n_features=n_features)

    # Run own implementation and scikit-learn version with the same parameters and compare the results
    brc = Birch(branching_factor=branching_factor, leaf_factor=branching_factor, threshold=threshold)
    brc.fit(X)
    sk_brc = sklearn_birch(branching_factor=branching_factor, threshold=threshold)
    sk_brc.fit(X)

    assert_array_equal(brc.all_centroids.shape, sk_brc.subcluster_centers_.shape)
