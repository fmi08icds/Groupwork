import logging
from pathlib import Path
from pandas import DataFrame
from collections.abc import Iterable

from regression_comparison.dataset.generate_data import generate_datasets
from regression_comparison.io import download_dataset, read_datasets

logger = logging.getLogger("regression_comparison")


def filter_hidden_files(file_list: Iterable[Path]) -> list[Path]:
    return [x for x in file_list if not x.name.startswith(".")]


def load_data(
    datasets_to_load: dict[str, str], non_generated: Path, generated: Path
) -> dict[str, DataFrame]:
    if not any(filter_hidden_files(non_generated.iterdir())):
        download_datasets(datasets_to_load, non_generated)
    if not any(filter_hidden_files(generated.iterdir())):
        generate_datasets(generated)

    file_paths = [
        x for x in filter_hidden_files(non_generated.parent.rglob("*")) if x.is_file()
    ]
    return read_datasets(file_paths)


def download_datasets(datasets_to_load: dict, non_generated: Path) -> None:
    logger.info("Download datasets")

    for dataset_name, url in datasets_to_load.items():
        download_dataset(url, non_generated, dataset_name)

    logger.info("Downloaded datasets")
