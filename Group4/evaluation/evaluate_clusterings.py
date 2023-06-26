import time

from numpy.typing import NDArray
from typing import Tuple
import os
import sys

# run with Groupwork/Group4 as current working directory

# append parent directory to sys.path, so that the "clustering" folder is found
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# print("\n".join(sys.path))

# self-implemented clustering algorithms, these DO NOT use scikit-learn
from clustering.affinity_propagation import affinity_propagation
from clustering.dbscan import dbscan

import data.datasource as ds

# load CPU acceleration for scikit-learn, see https://pypi.org/project/scikit-learn-intelex/
from sklearnex import patch_sklearn

patch_sklearn()

# all scikit-learn imports must be after the Intel patch
from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans, MiniBatchKMeans, BisectingKMeans
from sklearn.datasets import make_blobs
from sklearn import metrics


##########################################################
# Matrices as evaluation results
##########################################################

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.contingency_matrix.html
# metrics.cluster.contingency_matrix(labels_true, labels_pred)

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.pair_confusion_matrix.html
# metrics.cluster.pair_confusion_matrix(labels_true, labels_pred)


def get_dataset(n_centers, n_samples_per_genre, n_features, random_state) -> Tuple:
    X, labels_true = make_blobs(
        n_samples=n_centers * n_samples_per_genre, centers=n_centers, n_features=n_features, random_state=random_state
    )
    return X, labels_true


# runtime with sample size 100 => about 1 seconds
# runtime with sample size 1000 => about 5 seconds
# runtime with sample size 2000 => about 130 seconds (due to DBSCAN)


def get_spotify_dataset(n_samples_per_genre):
    X, y = ds.load_X_y(
        path="data/SpotifyFeatures.csv",
        sample_size=n_samples_per_genre,
        attribute_list=[
            "popularity",
            "acousticness",
            "danceability",
            # 'duration_ms',
            "energy",
            "instrumentalness",
            "liveness",
            "loudness",
            "speechiness",
            "tempo",
            "valence",
        ],
    )
    y = y.flatten()
    # get 1D array from 2D array, this is need for calculating clustering metrics
    return X, y


def get_pred_k_means(X, n_clusters, random_state):
    start_time = time.perf_counter()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit(X)
    end_time = time.perf_counter()
    return kmeans.labels_, end_time - start_time


def get_pred_minibatch_k_means(X, n_clusters, random_state):
    start_time = time.perf_counter()
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=2048, n_init="auto").fit(X)
    end_time = time.perf_counter()
    return kmeans.labels_, end_time - start_time


def get_pred_bisecting_k_means(X, n_clusters, random_state):
    start_time = time.perf_counter()
    kmeans = BisectingKMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
    end_time = time.perf_counter()
    return kmeans.labels_, end_time - start_time


def get_pred_affinity_propagation_gr4(X, random_state):
    start_time = time.perf_counter()
    damping = 0.7
    _, labels_pred = affinity_propagation(X=X, damping=damping, random_state=random_state)
    end_time = time.perf_counter()
    return labels_pred, end_time - start_time


def get_pred_affinity_propagation(X, random_state):
    start_time = time.perf_counter()
    damping = 0.7
    clustering = AffinityPropagation(damping=damping, random_state=random_state).fit(X)
    end_time = time.perf_counter()
    return clustering.labels_, end_time - start_time


def get_pred_dbscan_gr4(X, min_points, epsilon):
    start_time = time.perf_counter()
    labels_pred = dbscan(X=X, epsilon=epsilon, min_points=min_points)
    end_time = time.perf_counter()
    return labels_pred, end_time - start_time


def get_pred_dbscan(X, min_samples, epsilon):
    start_time = time.perf_counter()
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
    end_time = time.perf_counter()
    return clustering.labels_, end_time - start_time


def get_intrinsic_metrics(X: NDArray, labels_pred: NDArray):
    """Intrinsic metrics (do not need ground-truth labels), but need the datapoints to compute
    inter-cluster distances and intra-cluster distances"""
    silhouette_score = metrics.silhouette_score(X, labels_pred, metric="euclidean")
    calinski_harabasz_score = metrics.calinski_harabasz_score(X, labels_pred)
    davies_bouldin_score = metrics.davies_bouldin_score(X, labels_pred)
    return {
        "silhouette_score": silhouette_score,
        "calinski_harabasz_score": calinski_harabasz_score,
        "davies_bouldin_score": davies_bouldin_score,
    }


def get_extrinsic_metrics(labels_true: NDArray, labels_pred: NDArray):
    """Extrinsic metrics (need ground-truth labels and predicted labels), but need no data points"""

    # Rand index / Adjusted Rand Index (adjusts for chance)
    ri_score = metrics.rand_score(labels_true, labels_pred)
    ari_score = metrics.adjusted_rand_score(labels_pred, labels_true)

    # Mutual information / (Adjusted) Mutual Information / Normalized Mutual Information
    mi_score = metrics.mutual_info_score(labels_true, labels_pred)
    ami_score = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    nmi_score = metrics.normalized_mutual_info_score(labels_true, labels_pred)

    # V_Measure
    homo_score, comp_score, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)

    # Fowlkes-Mallows Score
    fm_score = metrics.fowlkes_mallows_score(labels_true, labels_pred)

    return {
        "rand_index_score": ri_score,
        "adj_rand_index_score": ari_score,
        "mut_info_score": mi_score,
        "adj_mut_info_score": ami_score,
        "norm_mut_info_score": nmi_score,
        "homogeneity_score": homo_score,
        "completeness_score": comp_score,
        "v_measure": v_measure,
        "fowlkes_mallows_score": fm_score,
    }


def main() -> None:
    random_state = 42
    n_centers = 25  # with more than 100 samples from spotify dataset, 1 genre is left out
    n_samples_per_genre = 1000

    # use a dataset of blobs for evaluating the clustering algorithms
    # n_features = 10
    # X, labels_true = get_dataset(n_centers, n_samples_per_genre, n_features, random_state)

    # alternatively: use the Spotify song dataset for evaluating the clustering algorithms
    X, labels_true = get_spotify_dataset(n_samples_per_genre=n_samples_per_genre)

    start_time = time.perf_counter()

    labels_pred_km, run_time = get_pred_k_means(X, n_centers, random_state)
    print("K-Means")
    print(f"Runtime: {run_time:0.4f} seconds")
    print(get_extrinsic_metrics(labels_true, labels_pred_km))
    # print(get_intrinsic_metrics(X, labels_pred_km))

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.pair_confusion_matrix.html

    print("Pair Confusion matrix:", metrics.cluster.pair_confusion_matrix(labels_true, labels_pred_km))
    contingency_matrix = metrics.cluster.contingency_matrix(labels_true, labels_pred_km)
    print("Contingency matrix:", contingency_matrix)

    labels_pred_mbkm, run_time = get_pred_minibatch_k_means(X, n_centers, random_state)
    print("MiniBatch K-Means")
    print(f"Runtime: {run_time:0.4f} seconds")
    print(get_extrinsic_metrics(labels_true, labels_pred_mbkm))
    # print(get_intrinsic_metrics(X, labels_pred_mbkm))

    labels_pred_bskm, run_time = get_pred_bisecting_k_means(X, n_centers, random_state)
    print("Bisecting K-Means")
    print(f"Runtime: {run_time:0.4f} seconds")
    print(get_extrinsic_metrics(labels_true, labels_pred_bskm))
    # print(get_intrinsic_metrics(X, labels_pred_bskm))

    # for make_blobs
    # labels_pred_dbs, run_time = get_pred_dbscan(X, min_samples=n_samples_per_genre//2, epsilon=6)
    # for Spotify dataset
    labels_pred_dbs, run_time = get_pred_dbscan(X, min_samples=n_samples_per_genre//20, epsilon=.175)
    print("DBSCAN (Scikit-Learn)")
    print(f"Runtime: {run_time:0.4f} seconds")
    print(get_extrinsic_metrics(labels_true, labels_pred_dbs))
    # print(get_intrinsic_metrics(X, labels_pred_dbs))

    print("Pair Confusion matrix:", metrics.cluster.pair_confusion_matrix(labels_true, labels_pred_dbs))
    contingency_matrix = metrics.cluster.contingency_matrix(labels_true, labels_pred_dbs)
    print("Contingency matrix:", contingency_matrix)

    ####################################################
    # Only run DBSCAN (Group4) with small sample size like 1000 samples per genre!
    # takes about 80 seconds
    ####################################################
    # for make_blobs
    # labels_pred_dbs, run_time = get_pred_dbscan_gr4(X, min_points=n_samples_per_genre//2, epsilon=6)
    # for Spotify dataset
    # labels_pred_dbs, run_time = get_pred_dbscan_gr4(X, min_points=n_samples_per_genre//20, epsilon=0.175)
    # print("DBSCAN (Group 4)")
    # print(f"Runtime: {run_time:0.4f} seconds")
    # print(get_extrinsic_metrics(labels_true, labels_pred_dbs))
    # # print(get_intrinsic_metrics(X, labels_pred_dbs))

    ####################################################
    # Only run Affinity Propagation with small sample size like 100 samples per genre!
    ####################################################
    # labels_pred_afp, run_time = get_pred_affinity_propagation(X, random_state)
    # print("Affinity Propagation (Scikit-Learn)")
    # print(f"Runtime: {run_time:0.4f} seconds")
    # print(get_extrinsic_metrics(labels_true, labels_pred_afp))
    # # print(get_intrinsic_metrics(X, labels_pred_afp))

    # labels_pred_afp, run_time = get_pred_affinity_propagation_gr4(X, random_state)
    # print("Affinity Propagation (Group 4)")
    # print(f"Runtime: {run_time:0.4f} seconds")
    # print(get_extrinsic_metrics(labels_true, labels_pred_afp))
    # # print(get_intrinsic_metrics(X, labels_pred_afp))

    end_time = time.perf_counter()
    print(f"Total runtime: {end_time - start_time:0.4f} seconds")


if __name__ == "__main__":
    main()
