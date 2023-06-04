import pytest
import numpy as np
from numpy.testing import assert_array_equal

from clustering.utils import distance_to_reference


def test_distance_to_reference_exception_handling():
    n_samples, n_features = 20, 3
    X = np.random.normal(size=(n_samples, n_features))
    with pytest.raises(ValueError):
        distance_to_reference(X, X)  # Invalid shape (second argument must be one-dimensional)
    with pytest.raises(ValueError):
        distance_to_reference(X, np.random.normal(size=n_features + 1))  # Incompatible shapes


def test_distance_to_reference_shape():
    n_samples, n_features = 20, 3
    X = np.random.normal(size=(n_samples, n_features))
    Y = np.random.normal(size=n_features)
    assert distance_to_reference(X, Y).shape == (n_samples,)


def test_distance_to_reference_values():
    X = np.array([[3, 3], [0, 0], [2, 0]])
    Y = np.array([0, -1])
    computed = distance_to_reference(X, Y)
    expected = [5, 1, np.sqrt(5)]
    assert_array_equal(computed, expected)
