from importlib import import_module
from typing import Callable, Sequence

from data.datasource import load_X_y

CONFIG = {
    "dbscan": [{"epsilon": 0.5, "min_points": 5}],
    "affinity_propagation": [{"damping": 0.5, "max_iter": 200, "convergence_iter": 15}],
}


def get_clustering_function(m_name: str = "dbscan", f_name: str = None) -> Callable:
    """Import a clustering function by its name

    Parameters:
        m_name: Name of the clustering module
        f_name: Name of the function in the module. If None, use m_name

        Returns: The specified clustering function
    """
    if f_name is None:
        f_name = m_name
    return getattr(import_module(f"clustering.{m_name}"), f_name)


def main(path: str = "./data/SpotifyFeatures.csv", algorithms: Sequence[str] = None, sample_size: int = 500):
    """Run experiments on the different clustering algorithms using predefined parameter dictionaries
    Parameters:
        path (str): Path to the CSV file containing Spotify track data.
        algorithms (Sequence[str]): List of algorithm names to be tested (must be keys of CONFIG dictionary).
        sample_size (int): Number of samples per genre.
    """

    if algorithms is None:
        algorithms = CONFIG.keys()

    X, y = load_X_y(path=path, sample_size=sample_size)
    for alg_name in algorithms:
        print("\n" + "-" * 50 + f"\n{alg_name}\n" + "-" * 50)
        param_dicts = CONFIG[alg_name]
        func = get_clustering_function(alg_name)

        for param_dict in param_dicts:
            center_indices, labels = func(X=X, **param_dict)

            # TODO: Replace printing with evaluation;
            #  Store results in dataframe with columns [sample_size, algorithm, parameters, score1, score2, ...]
            print(f"- Parameters: {param_dict}")
            print(f"- Number of clusters: {len(center_indices)}")
            print(f"- Center Indices: \n{center_indices}\n")
