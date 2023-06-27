""" This class is to be used to let the user interactively change the coefficients and the intercept of a regression model. """
import numpy as np
import matplotlib.pyplot as plt


class InteractiveRegression:
    """
    This class is used to let the user interactively change the coefficients and the intercept of a regression model.
    """

    # init method or constructor
    def __init__(self, x_data, y_data, b0, b1, b2):
        self.x_data = x_data
        self.y_data = y_data
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.predicted_values = None
        self.residuals = None

    # method for calculating the predicted values
    def calc_predicted_values(self):
        self.predicted_values = (
            self.b0 + self.b1 * self.x_data + self.b2 * self.x_data**2
        )
        return self.predicted_values

    # method for calculating the residuals
    def calc_residuals(self):
        self.residuals = self.y_data - self.predicted_values
        return self.residuals

    # method for calculating the sum of squares
    def calc_sum_of_squares(self):
        return round(np.sum(self.residuals**2), 2)

    # method for calculating the mean squared error
    def calc_mean_squared_error(self):
        return round(np.mean(self.residuals**2), 2)

    # method for calculating the root mean squared error
    def calc_root_mean_squared_error(self):
        return round(np.sqrt(self.calc_mean_squared_error()), 2)
