import urllib.request
import zipfile
import os


## 1. Real Estate Valuation Data Set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx"
filename = "./Groupwork/Group3/data/realworld/real_estate.xlsx"
urllib.request.urlretrieve(url, filename)


## 2. Wine Quality Data Set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
filename = "./Groupwork/Group3/data/realworld/winequality-red.csv"
urllib.request.urlretrieve(url, filename)

## 3. Air Quality Data Set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
filename = "./Groupwork/Group3/data/realworld/AirQualityUCI.zip"
urllib.request.urlretrieve(url, filename)

# unpack air quality data
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall("./Groupwork/Group3/data/realworld/")

## remove zip file
os.remove(filename)

