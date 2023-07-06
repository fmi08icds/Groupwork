"""This module implements the BaseRegression class"""
import numpy as np


class BaseRegression:
    """
    This class implements the base regression model that serves as a superclass
    with basic functionality for our concrete regression models.

    """

    def __init__(self, data, transposed, name):
        if transposed:
            data = np.transpose(data)
        self.name = name
        self.x_data = data[:, :-1]
        self.y_data = data[:, -1]
        self.predicted_values = None

    def get_x_column(self, i):
        """
        Retrieves the data of the independent variable of the sample of the
        given column index

        :param i: The column index of the independent variable
        :return: The associated data of the independent variable
        """
        return self.x_data[:, i]

    def get_sum_of_squares(self):
        """_summary_

        :return: _description_
        """
        return np.sum(
            (self.predicted_values[i] - self.y_data[i]) ** 2
            for i in range(len(self.y_data))
        )

    def get_mse(self):
        """
        Calculates the mean square root error of the currently fitted model.

        :return: The mean square root error of the currently fitted model
        """
        return self.get_sum_of_squares() / len(self.y_data)

    def get_mae(self):
        """
        Calculates the mean average error of the currently fitted model.

        :return: The mean average error of the currently fitted model
        """
        return np.sum(
            abs(self.predicted_values[i] - self.y_data[i])
            for i in range(len(self.y_data))
        ) / len(self.y_data)
