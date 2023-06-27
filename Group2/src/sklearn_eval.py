import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from smo_svm import SVM
from svm_train_hard_margin import SVM as SVM_HM
from eval import Eval


def standard_scaler(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    scaled_data = (data - mean) / std
    return scaled_data


def preprocess_data(data):
    data.replace("Female", 0, inplace=True)
    data.replace("Male", 1, inplace=True)
    data.replace("Other", 2, inplace=True)
    data.replace("No Info", 0, inplace=True)
    data.replace("not current", 2, inplace=True)
    data.replace("current", 1, inplace=True)
    data.replace("former", 3, inplace=True)
    data.replace("never", 4, inplace=True)
    data.replace("ever", 5, inplace=True)

    # Split the data into features (X) and target variable (y)
    X = data.drop('diabetes', axis=1)
    y = data['diabetes']

    # scale the data
    X = standard_scaler(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test