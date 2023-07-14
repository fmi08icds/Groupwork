"""This module implements the  LocallyWeightedRegression class"""
import math

import numpy as np
import inspect

from regression_edu.models.base_regression import BaseRegression


class LocallyWeightedRegression(BaseRegression):
    """
    This is class implements a locally weighted regression.
    """

    def __init__(
        self,
        data,
        transposed=False,
        name="",
        tau=0.5,
        sections=None,
        sigma=1,
    ):
        """
        Calculates the locally weighted regression for the given data.

        :param data: data is a Nx(d+1) matrix with the last column being Y and X being
            Nxd. data[0] accesses therefor the first sample.
        :param transposed: If the data is transposed and data[0] returns a vector
            of the first dimension of the samples.
        :param name: Optional parameter to pass a name.
        :param tau: Parameter to control how much the points in proximity should by
            weighted. A higher value means that more data points are considered and
            weighted higher.
        :param sections: Controls how many local regressions should be controlled.
            The default is per data point a regression.
        :param sigma: Controls what the variance of the gaussian function should be that
            is used to smooth the functions.
        """
        super().__init__(data, transposed, name)
        self.sigma = sigma if sigma is not None else 1
        if tau is None:
            tau = 0.5

        # removes y-column and adds column of ones for the bias (w0)
        x_data = np.asarray([np.insert(sample, 0, 1) for sample in self.x_data])
        # raise Exception
        def w(i):
            return [
                math.e ** (-((np.linalg.norm(x_data[i] - x_j)) ** 2) / (2 * tau**2))
                for x_j in x_data
            ]

        if sections is None and len(data) >= 100:
            sections = 100
        if sections is None or sections <= 0:
            weight_matrix = np.zeros((len(x_data), len(x_data), len(x_data)))
            for i in range(len(weight_matrix)):
                weight_matrix[i] = np.diag(w(i))
            self.centres = x_data[:, 1]
            
        else:
            x_sorted = x_data[x_data[:, 1].argsort()]
            weight_matrix = np.zeros((sections, len(x_data), len(x_data)))
            sec_len = len(x_sorted) // sections
            prev_indices = 0
            self.centres = []
            for i in range(sections):
                sec = x_sorted[i * sec_len : (1 + i) * sec_len][:, 1]
                self.centres.append(np.mean(sec))
                index = len(sec) // 2 + prev_indices
                weight_matrix[i] = np.diag(w(index))
                prev_indices += len(sec)
            self.centres = np.asarray(self.centres)

        def get_coefficient(i):
            return (
                np.linalg.pinv(np.transpose(x_data) @ weight_matrix[i] @ x_data)
                @ np.transpose(x_data)
                @ weight_matrix[i]
                @ self.y_data
            )

        self.coefficients = [get_coefficient(i) for i in range(len(weight_matrix))]
        self.predicted_values = np.asarray([self.predict(xi) for xi in x_data[:, 1:]])
    
    def gauss(self, centre, x, sigma):
        """
        Calculates the corresponding value of a given x using a gaussian distribution
        that is specified by the given center and sigma.

        :param centre: the expected value of the distribution
        :param x: The value where the distribution should be evaluated
        :param sigma: The variance of the distribution
        :return: The value of the gaussian function on x
        """
        return math.e ** (-((centre - x) ** 2) / (2 * sigma**2))

    def predict(self, x):
        """
        Calculates the prediction for a given datapoint. It doesn't support the input of
        multiple data points.

        :param x: A numeric value or vector consisting of one value for each factor of
            the data point
        :return: returns a float as the prediction for the given data point.
        """
        if not isinstance(x, np.ndarray):
            if isinstance(x, (int, float)):
                x = [x]
            x = np.asarray(x)
        length = len(x) if np.ndim(x) != 0 else 1
        if length != len(self.coefficients[0]) - 1:
            raise ValueError(
                f"x has to have the same dimension as x_data. dim x: {length}; \
dim x_data: {len(self.x_data[0])}"
            )

        summed = 0
        summed_gauss = sum(
            self.gauss(self.centres[index], x, self.sigma)
            for index in range(len(self.coefficients))
        )
        for index, coefficient in enumerate(self.coefficients):
            summed += (
                self.gauss(self.centres[index], x, self.sigma)
                / summed_gauss
                * sum(coefficient * np.insert(x, 0, 1))
            )

        # sometimes returned nested array. Should be fixed but this will extract
        # and return the actual value
        while True:
            try:
                summed = summed[0]
            except IndexError:
                return summed
