from sklearn.svm import SVC as SVM_SKL
from svm_train_hard_margin import SVM as SVM_HM
from smo_svm import SVM as SVM_SMO


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

        self.accuracy = round(correct / total, 4)

    def precision_score(self):
        true_positives = 0
        false_positives = 0

        for true, pred in zip(self.y, self.predictions):
            if pred == 1 and true == 1:
                true_positives += 1
            elif pred == 1 and true == 0:
                false_positives += 1

        self.precision = round(true_positives / (true_positives + false_positives + .1), 4)

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
        self.f1 = round(2 * (self.precision * self.recall) / (self.precision + self.recall + .1), 4)

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


def train_svm_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    if model == "HardMargin":
        svm = SVM_HM()
        svm.fit(X_train.to_numpy(), y_train.to_numpy())
    elif model == "SMO":
        # Change the ys from 0 to -1 
        y_train = y_train.to_numpy()
        y_train[y_train == 0] = -1
        y_val = y_val.replace(0, -1)
        y_test = y_test.replace(0, -1)

        svm = SVM_SMO()
        svm.fit(X_train.to_numpy(), y_train)
    elif model == "Sklearn":
        svm = SVM_SKL(kernel="poly") # best kernel based on sklearn_kernel_comparison.py
        svm.fit(X_train, y_train)

    print(model)
    val_predictions = svm.predict(X_val)
    ValidEval = Eval("Validation", y_val, val_predictions)
    ValidEval.get_eval_metrics()

    # Make predictions on the test set
    test_predictions = svm.predict(X_test)

    # Calculate evaluation metrics on the test set
    TestEval = Eval("Test", y_test, test_predictions)
    TestEval.get_eval_metrics()



