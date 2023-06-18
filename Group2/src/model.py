import numpy as np

class SVM:
    def __init__(self, learning_rate=0.01, num_iterations=1000, C=1.0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        # C: regularization paramter
        self.C = C
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # Init weights and biases
        self.weights = np.zeros(num_features)
        self.bias = 0.0

        for _ in range(self.num_iterations):
            for i in range(num_samples):
                if y[i] * (np.dot(X[i], self.weights) - self.bias) >= 1:
                    self.weights -= self.learning_rate * (2 * self.C * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.C * self.weights - np.dot(X[i], y[i]))
                    self.bias -= self.learning_rate * y[i]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)
    
    def hyperplane(self, x_0):
        return -(self.weights[0] * x_0 + self.bias) / self.weights[1]


# Some code to test the model
# Does not behave as expected yet
if __name__ == "__main__":
    # Dummy data
    X = np.array([[1, 1], [2, 1], [3, 2], [2, 3], [3, 3], [1, 3]])
    y = np.array([-1, -1, -1, 1, 1, 1])

    from matplotlib import pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

    svm = SVM()
    svm.fit(X, y)
    
    xs = np.linspace(1., 3.)
    ys = svm.hyperplane(xs)
    plt.plot(xs, ys)
    plt.show()

    new_data = np.array([[4, 3], [1, 2]])
    predictions = svm.predict(new_data)
    print(predictions)
