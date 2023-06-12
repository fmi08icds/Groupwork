import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def lib_pca():
    data = pd.read_csv('dataset.csv')
    pca = PCA(n_components=2)  # Anzahl der gewünschten Hauptkomponenten
    principal_components = pca.fit_transform(data)
