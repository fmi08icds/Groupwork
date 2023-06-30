from datetime import datetime
from itertools import product
from pandas import DataFrame
from sklearn import metrics
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import Callable, Sequence

from data.datasource import load_X_y
from clustering import dbscan, affinity_propagation, birch

import numpy as np
import sys
import time

np.random.seed(42)

def birch_wrapper(X, **kwargs):
    model = birch.Birch(**kwargs)
    model.fit(X)
    return None, model.labels

def kmeans_wrapper(X, **kwargs):
    model = KMeans(n_init="auto", **kwargs)
    model.fit(X)
    return None, model.labels_

CONFIG = {
    "Affinity Propagation": {
        "func": affinity_propagation.affinity_propagation,
        "params": {"damping": 0.7, "convergence_iter": 20, "max_iter": 200}
    },
    "DBSCAN": {
        "func": dbscan.dbscan,
        "params": {"epsilon": 0.3244, "min_points": 20}
    },
    "BIRCH": {
        "func": birch_wrapper,
        "params": {"branching_factor": 50, "threshold": 0.25, "n_cluster": 25, "predict": True}
    },
    "k-Means": {
        "func": kmeans_wrapper,
        "params": {"n_clusters": 25}
    }
}


def main(path: str = "./data/SpotifyFeatures.csv", algorithms: Sequence[str] = None, sample_size: int = 100):
    """Run experiments on the different clustering algorithms using predefined parameter dictionaries
    Parameters:
        path (str): Path to the CSV file containing Spotify track data.
        algorithms (Sequence[str]): List of algorithm names to be tested (must be keys of CONFIG dictionary).
        sample_size (int): Number of samples per genre.
    """

    if algorithms is None:
        algorithms = CONFIG.keys()

    results = []

    X, y, _ = load_X_y(path=path, sample_size=sample_size)
    print(y.shape)

    # Run clustering algorithms with their best parameters
    for alg_name in algorithms:
        params = CONFIG[alg_name]["params"]
        func = CONFIG[alg_name]["func"]

        # Measure runtime
        start_time = time.time()
        _, labels = func(X=X, **params)
        duration = time.time() - start_time

        # Compute evaluation results
        results.append({
            "Score": metrics.adjusted_rand_score(labels_true=y[:, 0], labels_pred=labels),
            "Algorithm": alg_name,
            "Parameters": "<br>".join(f"`{k} = {v}`" for k, v in params.items()),
            "# Clusters": np.max(labels) + 1,
            "Runtime": f"{duration * 1000:.1f} ms"
        })

    df = DataFrame(results)
    # Rank by score
    df.sort_values("Score", inplace=True, ascending=False)
    # Print dataframe as a markdown table to stdout
    df.to_markdown(sys.stdout, index=False)
    print()

if __name__ == "__main__":
    main()