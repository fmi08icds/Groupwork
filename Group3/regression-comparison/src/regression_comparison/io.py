import logging
from pathlib import Path
import pandas as pd
import urllib.request
import zipfile

logger = logging.getLogger("regression_comparison")


def save_results(df: pd.DataFrame, path: Path):
    df.to_csv(str(path), index=False)
    logger.info(f"Results saved to {path}")


def read_datasets(file_paths: list[Path]) -> dict[str, pd.DataFrame]:
    datasets = {}

    for file_path in file_paths:
        separator = ";" if file_path.stem == "wine_quality" else ","
        index_col = "No" if file_path.stem == "real_estate" else None
        if file_path.suffix == ".xlsx":
            datasets[file_path.stem] = pd.read_excel(file_path, index_col=index_col)
        elif file_path.suffix == ".csv":
            datasets[file_path.stem] = pd.read_csv(file_path, sep=separator)

    return datasets


def download_dataset(url: str, non_generated: Path, dataset_name: str) -> None:
    filename = Path(non_generated, dataset_name).with_suffix(Path(url).suffix)
    urllib.request.urlretrieve(url, str(filename))

    # Handle extraction of zipped dataset
    if dataset_name == "air_quality":
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(non_generated)
            filename.unlink()
            filename = filename.with_name(Path(url).stem)
            filename.with_suffix(".csv").unlink()
            filename.with_suffix(".xlsx").replace(
                filename.with_name(dataset_name).with_suffix(".xlsx")
            )
