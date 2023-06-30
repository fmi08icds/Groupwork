from datetime import datetime
from importlib import import_module
from itertools import product
from pandas import DataFrame
from sklearn import metrics
from tqdm import tqdm
from typing import Callable, Sequence

from data.datasource import load_X_y

aff_damping = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
aff_max_iter = [200, 400]
aff_convergence_iter = [10, 20]

CONFIG = {
    "affinity_propagation": [{"damping": x[0], "max_iter": x[1], "convergence_iter": x[2]}
                             for x in product(aff_damping, aff_max_iter, aff_convergence_iter)],
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

    results = {'algorithm': [], 'parameters': [], 'ari_score': [], 'num_clusters': []}

    X, y, _ = load_X_y(path=path, sample_size=sample_size)
    for alg_name in algorithms:
        param_dicts = CONFIG[alg_name]
        func = get_clustering_function(alg_name)

        for param_dict in tqdm(param_dicts, desc=f'Running configurations for {alg_name}'):
            center_indices, labels = func(X=X, **param_dict)
            results['algorithm'].append(alg_name)
            results['parameters'].append(param_dict)
            results['ari_score'].append(metrics.adjusted_rand_score(labels_true=y[:, 0], labels_pred=labels))
            results['num_clusters'].append(len(center_indices))

    DataFrame(results).to_csv(f'experiments_{datetime.now().strftime("%d%m_%H-%M")}.csv')

if __name__ == '__main__':
    main()