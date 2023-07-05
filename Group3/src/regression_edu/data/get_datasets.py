# dependencies: imports + openpyxl (for converting to excel)
import urllib.request
import os
import pandas as pd

url_base = "./realworld/"

## 1. Real Estate Valuation Data Set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx"
filename = url_base + "real_estate.xlsx"
urllib.request.urlretrieve(url, filename)


## 2. Wine Quality Data Set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
filename = url_base + "winequality-red.csv"
urllib.request.urlretrieve(url, filename)

# ## 3. Air Quality Data Set
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
# filename = "./Groupwork/Group3/data/realworld/AirQualityUCI.zip"
# urllib.request.urlretrieve(url, filename)
#
# # unpack air quality data
# with zipfile.ZipFile(filename, 'r') as zip_ref:
#     zip_ref.extractall("./Groupwork/Group3/data/realworld/")
#
# ## remove zip file
# os.remove(filename)
#
# ## remove duplicate
# os.remove("./Groupwork/Group3/data/realworld/AirQualityUCI.csv")

## 4. prostate
url = "https://hastie.su.domains/ElemStatLearn/datasets/prostate.data"
filename = url_base + "prostate.data"
urllib.request.urlretrieve(url, filename)

## convert .data to .xlsx
df = pd.read_csv(url_base + "prostate.data", sep='\t')
df.to_excel(url_base + "prostate.xlsx", index=False)
os.remove(filename)



