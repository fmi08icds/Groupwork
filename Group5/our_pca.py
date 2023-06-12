from pandas import DataFrame
import numpy as np


def our_pca(data: DataFrame, n_components):
    # centering the data
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    # Computing the covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sorting of the eigenvalues and vectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]

    # Selecting the principal components
    selected_eigenvectors = sorted_eigenvectors[:, :n_components]

    # Transforming the data
    transformed_data = np.dot(centered_data.transpose(), selected_eigenvectors)

    return transformed_data

