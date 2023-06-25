import math

import numpy as np


class linear_regression:
    coeffs = None
    predicted_values = None
    x_data = None
    y_data = None

    def __init__(self, data, transposed=False):
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
        self.x_data = np.asarray([np.insert(sample[:-1], 0, 1) for sample in data])

        self.coeffs = np.linalg.pinv(np.transpose(self.x_data) @ self.x_data) @ np.transpose(self.x_data) @ self.y_data
        self.predicted_values = self.x_data @ self.coeffs

    def get_coeffs(self):
        # global coeffs
        return self.coeffs

    def get_predicted_values(self):
        # global predicted_values
        return self.predicted_values

    def get_sum_of_squares(self):
        return np.sum([(self.predicted_values[i] - self.y_data[i]) ** 2 for i in range(len(self.y_data))])

    def get_x_data(self):
        return self.x_data[:, 1:]

    def get_y_data(self):
        return self.y_data


class lwr:
    coeffs = None
    predicted_values = None
    x_data = None
    y_data = None
    W = None

    def __init__(self, data, transposed=False, tau=.5):
        """
        Calculates the ordinary linear regression for the given data.
        :param data: data is a Nx(d+1) matrix with the last column being Y and X being Nxd. data[0] accesses therefor the first sample.
        :param transposed: If the data is transposed and data[0] returns a vector of the first dimension of the samples.
        """
        data = np.asarray(data)
        if transposed:
            data = np.transpose(data)

        self.y_data = data[:, -1]
        # removes y-column and adds column of ones for the bias (w0)
        self.x_data = np.asarray([np.insert(sample[:-1], 0, 1) for sample in data])

        def w(i):
            return [math.e ** (-(np.linalg.norm(self.x_data[i] - x_j)) ** 2 / (2 * tau ** 2)) for x_j in self.x_data]

        W = np.zeros((len(self.x_data), len(self.x_data), len(self.x_data)))
        for i in range(len(W)):
            W[i] = np.diag(w(i))

        def get_coeffs_i(i):
            return np.linalg.pinv(np.transpose(self.x_data) @ W[i] @ self.x_data) @ np.transpose(self.x_data) @ W[i] @ self.y_data

        predicted_values_0 = self.x_data @ get_coeffs_i(0)
        predicted_values_1 = self.x_data @ get_coeffs_i(1)
        self.predicted_values = np.asarray(
            [predicted_values_1[i] if i < len(predicted_values_0) / 2 else predicted_values_1[i] for i in
             range(len(predicted_values_0))])
        self.predicted_values = np.asarray([(self.x_data @ get_coeffs_i(i))[i] for i in range(len(data))])

    def get_W(self):
        return self.W

    def get_coeffs(self):
        return self.coeffs

    def get_predicted_values(self):
        # global predicted_values
        return self.predicted_values

    def get_sum_of_squares(self):
        return np.sum([(self.predicted_values[i] - self.y_data[i]) ** 2 for i in range(len(self.y_data))])

    def get_x_data(self):
        return self.x_data[:, 1:]

    def get_y_data(self):
        return self.y_data
