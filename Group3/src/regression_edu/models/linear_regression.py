import math

import numpy as np

class LinearRegression:
    coeffs = None
    predicted_values = None
    x_data = None
    y_data = None
    name = ""

    # def __init__(self, data, transposed=False, name=""):
    #     self.calc(data, transposed, name)


    def __init__(self, data, transposed=False, name=""):
        """
        Calculates the ordinary linear regression for the given data.
        :param data: data is a Nx(d+1) matrix with the last column being Y and X being Nxd. data[0] accesses therefor the first sample.
        :param transposed: If the data is transposed and data[0] returns a vector of the first dimension of the samples.
        """
        data = np.asarray(data)
        if transposed:
            data = np.transpose(data)

        self.y_data = data[:, -1]
        # removes y-column and ads column of ones for the bias (w0)
        self.x_data = data[:,:-1]
        x_data = np.asarray([np.insert(sample, 0, 1) for sample in self.x_data])
        self.coeffs = np.linalg.pinv(np.transpose(x_data) @ x_data) @ np.transpose(x_data) @ self.y_data
        self.predicted_values = x_data @ self.coeffs
        self.name = ""

    def f(self, x):
        if type(x) is not np.array(()):
            if type(x) in [int, float]:
                x = [x]
            x = np.asarray(x)
        length = len(x) if np.ndim(x) != 0 else 1
        if length != len(self.coeffs) - 1:
            raise ValueError(
                f"x has to have the same dimension as x_data. dim x: {len(x)}; dim x_data: {len(self.x_data[0])}")
        return sum(self.coeffs * np.insert(x, 0, 1))

    def get_x_column(self, i):
        return self.x_data[:,i]

    def get_sum_of_squares(self):
        return np.sum([(self.predicted_values[i] - self.y_data[i]) ** 2 for i in range(len(self.y_data))])


    def get_MSE(self):
        return self.get_sum_of_squares()/len(self.y_data)


    def get_MAE(self):
        return np.sum([abs(self.predicted_values[i] - self.y_data[i]) for i in range(len(self.y_data))]) / len(self.y_data)