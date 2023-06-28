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

    # Computing the covariance matrix in smaller steps
    # n_samples, n_features = data.shape
    # covariance_matrix = np.zeros((n_features, n_features))
    # for i in range(n_samples):
    #     covariance_matrix += np.outer(data[i], data[i])
    # covariance_matrix /= n_samples

    # eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sorting the eigenvalues and vectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]

    # Selecting the principal components
    selected_eigenvectors = sorted_eigenvectors[:, :n_components].real
    # Selecting the eigenvalues
    selected_eigenvalues = sorted_eigenvalues[:n_components].real

    # Transforming the data
    # transformed_data = np.dot(centered_data, selected_eigenvectors)

    return selected_eigenvalues, selected_eigenvectors


def apply_components(data, eigenvectors):
    """
    Apply the eigenvectors to data (project the data into the eigen-space.
    :param data: data to be projected into the eigen-space
    :param eigenvectors: components (eigenvectors from PCA)
    :return: projected data
    """
    eigen_matrix = np.array(eigenvectors)
    if data.shape[1] != eigen_matrix.shape[0]:
        eigen_matrix = eigen_matrix.transpose()
    # Transforming the data
    transformed_data = np.matmul(data, eigen_matrix)

    return transformed_data
