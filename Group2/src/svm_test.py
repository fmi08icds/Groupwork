import time
import pandas as pd
from eval import train_svm_model
from sklearn_eval import preprocess_data  ####### Replace with preprocessing DataAnalyzer as soon as it's finished


def measure_execution_time(function, *args):
    start_time = time.time()
    function(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")


def main():
    data = pd.read_csv('../data/diabetes_prediction_dataset.csv')
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data)

    measure_execution_time(train_svm_model, "Sklearn", X_train, X_val, X_test, y_train, y_val, y_test)
    #measure_execution_time(train_svm_model, "SMO", X_train, X_val, X_test, y_train, y_val, y_test)
    #measure_execution_time(train_svm_model, "HardMargin", X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    main()
