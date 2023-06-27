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

    def f(self, x):
        x = np.asarray([x])
        if len(x) != len(self.coeffs) - 1:
            raise ValueError(
                f"x has to have the same dimension as x_data. dim x: {len(x)}; dim x_data: {len(self.get_x_data()[0])}")
        return sum(self.coeffs * np.insert(x, 0, 1))

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
            return np.linalg.pinv(np.transpose(self.x_data) @ W[i] @ self.x_data) @ np.transpose(self.x_data) @ W[
                i] @ self.y_data

        avg = np.average(self.x_data[:1])
        x_data_left = np.empty(shape=(0, len(self.x_data[0])))
        x_data_right = np.empty(shape=(0, len(self.x_data[0])))
        for data in self.x_data:
            if data[1] <= avg:
                x_data_left = np.append(x_data_left, [data], axis=0)
            else:
                x_data_right = np.append(x_data_right, [data], axis=0)
        self.centres = self.x_data[:,1]
        self.coeffs = [get_coeffs_i(i) for i in range(len(self.x_data))]
        self.predicted_values = np.asarray([self.f(xi) for xi in self.x_data[:, 1:]])

    def f(self, x):
        if type(x) is not np.array(()):
            if type(x) in [int, float]:
                x = [x]
            x = np.asarray([x])
        if len(x) != len(self.coeffs[0]) - 1:
            raise ValueError(
                f"x has to have the same dimension as x_data. dim x: {len(x)}; dim x_data: {len(self.get_x_data()[0])}")

        def gauss(centre, x, sigma=1): return math.e ** (-(centre - x) ** 2 / (2 * sigma ** 2))
        summed = 0
        summed_gauss = sum([gauss(self.centres[index], x) for index in range(len(self.coeffs))])
        for index, coeff in enumerate(self.coeffs):
            summed += gauss(self.centres[index], x)/ summed_gauss * sum(coeff * np.insert(x, 0, 1))
        while True:
            try:
                summed = summed[0]
            except IndexError:
                return summed

    def get_centres(self):
        return self.centres

    def get_W(self):
        return self.W

    def get_coeffs(self):
        return self.coeffs

    def get_predicted_values(self):
        return self.predicted_values

    def get_sum_of_squares(self):
        return np.sum([(self.predicted_values[i] - self.y_data[i]) ** 2 for i in range(len(self.y_data))])

    def get_x_data(self):
        return self.x_data[:, 1:]

    def get_y_data(lmaojustforthelols):
        return lmaojustforthelols.y_data
