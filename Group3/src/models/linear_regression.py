
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
        :param data: data is a Nx(d+1) matrix with the last column being Y and 
                    X being Nxd. data[0] accesses therefor the first sample.
        :param transposed: If the data is transposed and data[0] returns a
                    vector of the first dimension of the samples.
        """
        data = np.asarray(data)
        if transposed:
            data = np.transpose(data)

        self.y_data = data[:, -1]
        # removes y-column and ads column of ones for the bias (w0)
        self.x_data = np.asarray([np.insert(sample[:-1], 0, 1) for sample in data])

        self.coeffs = (
            np.linalg.pinv(np.transpose(self.x_data) @ self.x_data)
            @ np.transpose(self.x_data)
            @ self.y_data
        )
        self.predicted_values = self.x_data @ self.coeffs

    def f(self, x):
        x = np.asarray([x])
        if len(x) != len(self.coeffs) - 1:
            raise ValueError(
                f"x has to have the same dimension as x_data. dim x: {len(x)}; dim x_data: {len(self.get_x_data()[0])}"
            )
        return sum(self.coeffs * np.insert(x, 0, 1))

    def get_coeffs(self):
        # global coeffs
        return self.coeffs

    def get_predicted_values(self):
        # global predicted_values
        return self.predicted_values

    def get_sum_of_squares(self):
        return np.sum(
            [
                (self.predicted_values[i] - self.y_data[i]) ** 2
                for i in range(len(self.y_data))
            ]
        )

    def get_x_data(self):
        return self.x_data[:, 1:]

    def get_y_data(self):
        return self.y_data
