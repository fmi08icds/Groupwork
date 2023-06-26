import pandas as pd
from eval import train_svm_model
from sklearn_eval import preprocess_data  ####### Replace with preprocessing DataAnalyzer as soon as its finished


def main():
    file = '../data/diabetes_prediction_dataset.csv'
    data = pd.read_csv(file)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data)  ####### Replace with preprocessing DataAnalyzer as soon as its finished
    #train_svm_model("Sklearn", X_train, X_val, X_test, y_train, y_val, y_test)
    #train_svm_model("SMO", X_train, X_val, X_test, y_train, y_val, y_test)
    train_svm_model("HardMargin", X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    main()