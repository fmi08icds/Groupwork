import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class Eval:
    def __init__(self, set_type, y, predictions):
        self.set_type = set_type
        self.y = y
        self.predictions = predictions
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.conf_matrix = None

    def get_eval_metrics(self):
        self.accuracy_score()
        self.precision_score()
        self.recall_score()
        self.f1_score()
        self.confusion_matrix()
        self.print_metrics()

    def accuracy_score(self):
        correct = 0
        total = len(self.y)

        for true, pred in zip(self.y, self.predictions):
            if true == pred:
                correct += 1

        self.accuracy = correct / total

    def precision_score(self):
        true_positives = 0
        false_positives = 0

        for true, pred in zip(self.y, self.predictions):
            if pred == 1 and true == 1:
                true_positives += 1
            elif pred == 1 and true == 0:
                false_positives += 1

        self.precision = true_positives / (true_positives + false_positives)

    def recall_score(self):
        true_positives = 0
        false_negatives = 0

        for true, pred in zip(self.y, self.predictions):
            if pred == 1 and true == 1:
                true_positives += 1
            elif pred == 0 and true == 1:
                false_negatives += 1

        self.recall = round(true_positives / (true_positives + false_negatives), 4)

    def f1_score(self):
        self.f1 = round(2 * (self.precision * self.recall) / (self.precision + self.recall), 4)

    def confusion_matrix(self):
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for true, pred in zip(self.y, self.predictions):
            if true == 1 and pred == 1:
                true_positives += 1
            elif true == 0 and pred == 0:
                true_negatives += 1
            elif true == 0 and pred == 1:
                false_positives += 1
            elif true == 1 and pred == 0:
                false_negatives += 1

        self.conf_matrix = [[true_negatives, false_positives],
                            [false_negatives, true_positives]]

    def print_metrics(self):
        print(self.set_type, "Accuracy:", self.accuracy)
        print(self.set_type, "Precision:", self.precision)
        print(self.set_type, "Recall:", self.recall)
        print(self.set_type, "F1 Score:", self.f1)
        print(self.set_type, "Confusion Matrix:")
        print('\n'.join([' '.join([str(item) for item in row]) for row in self.conf_matrix]))

# Load the dataset from CSV
data = pd.read_csv('../data/diabetes_prediction_dataset.csv')
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

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Create an SVM classifier
svm = SVC()

# Train the SVM classifier
svm.fit(X_train, y_train)

# Make predictions on the validation set
val_predictions = svm.predict(X_val)

# Calculate evaluation metrics on the validation set
ValidEval = Eval("Validation", y_val, val_predictions)
ValidEval.get_eval_metrics()

# Make predictions on the test set
test_predictions = svm.predict(X_test)

# Calculate evaluation metrics on the test set
TestEval = Eval("Test", y_test, test_predictions)
TestEval.get_eval_metrics()
