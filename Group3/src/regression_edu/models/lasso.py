"""This module implements the LassoRegression class"""
import numpy as np

from regression_edu.models.base_regression import BaseRegression


class LassoRegression(BaseRegression):
    """
    This class implements a lasso regression using ordinary least squares.
    Additionally it regularizes the parameters of the model.
    """

    def __init__(self, data, transposed=False, name=""):
        """
        Calculates the linear regression with regularization for the given data.
        :param data: data is a Nx(d+1) matrix with the last column being Y and
            X being Nxd. data[0] accesses therefor the first sample.
        :param transposed: If the data is transposed and data[0] returns a
            vector of the first dimension of the samples.
        :param name: Optional parameter to pass a name.
        """
        super().__init__(data, transposed, name)
        x_data = np.asarray([np.insert(sample, 0, 1) for sample in self.x_data])
        self.coefficients = (
            np.linalg.pinv(np.transpose(x_data) @ x_data)
            @ np.transpose(x_data)
            @ self.y_data
        )
        self.predicted_values = x_data @ self.coefficients

    def predict(self, x):
        """
        Calculates the prediction for a given data point. It doesn't support the input
        of multiple data points.
        :param x: A numeric value or vector consisting of one value for each factor
            of the data point
        :return: returns a float as the prediction for the given data point.
        """
        if not isinstance(x, np.ndarray):
            if type(x) in [int, float]:
                x = [x]
            x = np.asarray(x)
        length = len(x) if np.ndim(x) != 0 else 1
        if length != len(self.coefficients) - 1:
            raise ValueError(
                f"x has to have the same dimension as x_data. dim x: {len(x)}; \
                dim x_data: {len(self.x_data[0])}"
            )
        return sum(self.coefficients * np.insert(x, 0, 1))