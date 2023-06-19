from pandas import DataFrame
import numpy as np


def our_pca(data: DataFrame, n_components):
    """
    Own implementation of pca.
    :param data: pandas DataFrame
    :param n_components: number of components
    :return: (eigenvalues, eigenvectors)
    """
    # Computing the covariance matrix
    covariance_matrix = np.cov(data, rowvar=False)

    # eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sorting of the eigenvalues and vectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]

    # Selecting the principal components
    selected_eigenvectors = sorted_eigenvectors[:, :n_components]
    # Selecting the eigenvalues
    selected_eigenvalues = sorted_eigenvalues[:, :n_components]

    # Transforming the data
    # transformed_data = np.dot(centered_data, selected_eigenvectors)

    return selected_eigenvalues, selected_eigenvectors


def apply_components(data, eigenvalues):
    """
    Apply the eigenvalues to data (project the data into the eigen-space.
    :param data: data to be projected into the eigen-space
    :param eigenvalues: components (eigenvectors from PCA)
    :return: projected data
    """
    transformed_data = np.dot(data, eigenvalues)
    return transformed_data
