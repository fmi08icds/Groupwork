import pandas as pd
from sklearn.model_selection import train_test_split
from model import SVM 

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv("Group2/data/diabetes_prediction_dataset.csv")
    df = df.sample(100)
    df["diabetes"].replace(0, -1, inplace=True)

    train_df, test_df = train_test_split(df, test_size=0.2)

    # Split the data into labels and features
    train_labels, test_labels = train_df["diabetes"], test_df["diabetes"]
    train_features, test_features = train_df.drop("diabetes", axis=1), test_df.drop("diabetes", axis=1)

    # Use only some numeric columns for quick svm testing purposes
    sample_columns = ["bmi", "blood_glucose_level"]
    sample_test_features = test_features[sample_columns]
    sample_train_features = train_features[sample_columns]

    # Fit SVM
    svm = SVM()
    svm.fit(sample_train_features.to_numpy(), train_labels.to_numpy())

    predicted_test_labels = svm.predict(sample_test_features)
    print("predicted_test_labels", predicted_test_labels)
    print("test_labels", test_labels.to_numpy())
    print(pd.concat((test_labels, predicted_test_labels - test_labels), axis=1))
