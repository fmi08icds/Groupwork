# dependencies: imports + openpyxl (for converting to excel)
import urllib.request
import os
import pandas as pd

base_directory = "./realworld/"

## 1. Real Estate Valuation Data Set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx"
filename = base_directory + "real_estate.xlsx"
urllib.request.urlretrieve(url, filename)


## 2. Wine Quality Data Set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
filename = base_directory + "winequality-red.csv"
urllib.request.urlretrieve(url, filename)

## 3. prostate
url = "https://hastie.su.domains/ElemStatLearn/datasets/prostate.data"
filename = base_directory + "prostate.data"
urllib.request.urlretrieve(url, filename)

## convert .data to .xlsx
df = pd.read_csv(base_directory + "prostate.data", sep="\t")
df.to_excel(base_directory + "prostate.xlsx", index=False)
os.remove(filename)
