import logging
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd

from regression_comparison.dataset.generate_data import generate_datasets

logger = logging.getLogger("regression_comparison")


def filter_hidden_files(file_list: list):
    return [x for x in file_list if not x.name.startswith(".")]


def load_data(datasets_to_load):
    data = Path("data")
    non_generated = Path(data, "non_generated")
    generated = Path(data, "generated")

    if not any(filter_hidden_files(non_generated.iterdir())):
        download_datasets(datasets_to_load, non_generated)
    if not any(filter_hidden_files(generated.iterdir())):
        generate_datasets(generated)

    return read_data(data)


def read_data(data_path: Path) -> 'dict[str, pd.DataFrame]':
    files = [x for x in list(data_path.rglob("*")) if x.is_file() and x.name[0] != "."]
    datasets = {}

    for file in files:
        if file.suffix == ".xlsx":
            datasets[file.stem] = pd.read_excel(file)
        elif file.suffix == ".csv":
            datasets[file.stem] = pd.read_csv(file)

    return datasets


def download_datasets(datasets_to_load: dict, non_generated: Path) -> None:
    logger.info("Download datasets")

    for dataset_name, url in datasets_to_load.items():
        filename = Path(non_generated, dataset_name).with_suffix(Path(url).suffix)

        urllib.request.urlretrieve(url, str(filename))

        # Handle extraction of dataset
        if dataset_name == "air_quality":
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(non_generated)
                filename.unlink()
                filename = filename.with_name(Path(url).stem)
                filename.with_suffix(".csv").unlink()
                filename.with_suffix(".xlsx").replace(
                    filename.with_name(dataset_name).with_suffix(".xlsx")
                )

    logger.info("Downloaded datasets")
