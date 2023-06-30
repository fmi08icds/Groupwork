from sklearn.cluster import affinity_propagation as sklearn_affinity_propagation
from sklearn.datasets import make_blobs
from numpy.testing import assert_array_equal

from clustering.affinity_propagation import affinity_propagation, compute_similarity

import pytest


@pytest.mark.parametrize("n_centers", [1, 2, 4, 5])
@pytest.mark.parametrize("n_features", [2, 3, 5, 10])
@pytest.mark.parametrize("damping", [0.5, 0.7, 0.9])
@pytest.mark.parametrize("random_state", [1, 2])
def test_affinity_propagation_compare_results(n_centers: int, n_features: int, damping: float, random_state: int):
    # Generate dataset with `n_centers` blobs (circular clusters)
    X, _ = make_blobs(n_samples=n_centers * 10, centers=n_centers, n_features=n_features, random_state=random_state)

    # Run own implementation and scikit-learn version with the same parameters and compare the results
    computed_indices, computed_labels = affinity_propagation(X=X, damping=damping, random_state=random_state)

    expected_indices, expected_labels = sklearn_affinity_propagation(
        S=compute_similarity(X), damping=damping, random_state=random_state
    )

    # Confirm that the results are identical
    assert_array_equal(computed_indices, expected_indices)
    assert_array_equal(computed_labels, expected_labels)
