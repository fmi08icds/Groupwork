## generate easy data
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger("regression_comparison")


def generate_datasets(generated: Path) -> None:
    logger.info("Generate data")

    # simple linear y
    data_range = [-25, 25]
    y_true = range(data_range[0], data_range[1])
    size = len(y_true)

    # epsilon is normally distributed (easiest & standard case)
    mean = 0
    std = 500
    eps = np.random.normal(mean, std, size)

    vars = pd.DataFrame(
        {
            "x1": [(150 * x + 1.02 * x**2 + 10) for x in y_true],
            "x2": [(-500 * x + 1.05 * x**2 - 300) for x in y_true],
            "x3": [(225 * x) for x in y_true],
            "x4": [(15 * x) for x in y_true],
        }
    )

    ## generate easy data from polynomials

    y = vars["x1"] + vars["x2"] + vars["x3"] + vars["x4"] + eps

    # append y to vars
    vars["y"] = y
    # save data to csv
    vars.to_csv(Path(generated, "easy_polynomials.csv"), index=False)

    logger.info("Generated data")
