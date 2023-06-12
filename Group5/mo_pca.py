import pandas as pd
import numpy as np


def run_pca():
    data = pd.read_csv('standard_test_data.csv')
    n_components = 2
    pca(data, n_components)


def pca(data, n_components):
    # Zentrieren der Daten
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    # Berechnung der Kovarianzmatrix
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Eigenwertzerlegung der Kovarianzmatrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sortierung der Eigenwerte und -vektoren absteigend
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]

    # Auswahl der Hauptkomponenten
    selected_eigenvectors = sorted_eigenvectors[:, :n_components]

    # Transformation der Daten
    transformed_data = np.dot(centered_data, selected_eigenvectors)

    return transformed_data

