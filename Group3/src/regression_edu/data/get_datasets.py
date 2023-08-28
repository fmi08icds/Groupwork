import os
import urllib.request

import pandas as pd

BASE_DIR = "./realworld/"
## COMMENTS: Seen!
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

datasets = [
    {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx",
        "file_path": BASE_DIR + "real_estate.xlsx",
    },
    {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "file_path": BASE_DIR + "winequality-red.csv",
    },
    {
        "url": "https://hastie.su.domains/ElemStatLearn/datasets/prostate.data",
        "file_path": BASE_DIR + "prostate.csv",
    },
]

for dataset in datasets:
    urllib.request.urlretrieve(dataset["url"], dataset["file_path"])

## convert .csv to .xlsx
prostate_file_path = datasets[-1]["file_path"]
df = pd.read_csv(prostate_file_path, sep="\t")
df.to_excel(BASE_DIR + "prostate.xlsx", index=False)
os.remove(prostate_file_path)
