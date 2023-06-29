import pandas as pd
from evaluation.eval import train_svm_model
from data_preprocessing import DataAnalyzer
from evaluation.sklearn_kernel_comparison import *



def main():
    # Read the data
    file =  "/Users/abdulnaser/Desktop/Groupwork/Group2/data/diabetes_prediction_dataset.csv"

    # Preprocess and split the data
    data_analyzer = DataAnalyzer(file)
    data_analyzer.preprocessing()
    X_train, X_test, y_train,y_test = data_analyzer.data_split()
    train_svm_model("SMO", X_train, X_test, y_train, y_test)

    # Compare our Accuracy to the Accuracy of Sklearn
    print("The results of the sklearn library!")
    eval_sklearn_imp(X_train,X_test,y_train,y_test)




if __name__ == "__main__":
    main()

