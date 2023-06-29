from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def eval_sklearn_imp(X_train, X_test, y_train,y_test):
    # Hyperparameter für die Validierung festlegen
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    best_accuracy = 0
    best_kernel = ''

    # test different kernels on vlidation set
    for kernel in kernels:
        # init and train model
        svm = SVC(kernel=kernel)
        svm.fit(X_train, y_train)

        # predict on test set
        y_test_pred = svm.predict(X_test)

        # calculate accuracy on validation set
        accuracy = accuracy_score(y_test, y_test_pred)
        print(kernel, ": ",accuracy)

        # save best kernel
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_kernel = kernel

    print("Best Kernel:", best_kernel)
    print("Test Accuracy:", best_accuracy)



"""
# load the dataset and separate target from the rest:
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
X = data.drop('diabetes', axis=1)
y = data['diabetes']

#iris = datasets.load_breast_cancer()
#X = iris.data
#y = iris.target

# scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split dataset in train, val and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# Hyperparameter für die Validierung festlegen
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

best_accuracy = 0
best_kernel = ''

# test different kernels on vlidation set
for kernel in kernels:
    # init and train model
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)

    # predict on validation set
    y_val_pred = svm.predict(X_val)

    # calculate accuracy on validation set
    accuracy = accuracy_score(y_val, y_val_pred)
    print(kernel, ": ",accuracy)

    # save best kernel
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_kernel = kernel

# train with the best kernel
best_svm = SVC(kernel=best_kernel)
best_svm.fit(X_train, y_train)

# predict with the test set
y_test_pred = best_svm.predict(X_test)

# Calculate accuracy with the test set
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Best Kernel:", best_kernel)
print("Test Accuracy:", test_accuracy)

"""