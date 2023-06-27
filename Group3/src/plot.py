import matplotlib.pyplot as plt
import numpy as np


def plot_2d(reg, show_predictands=False, show_residuals=True):
    """
    Plot data with one dependent and one independent variable to show a 2D plot.
    :param reg: An instance of a regression from a class in regression.py
    :param show_predictands: By default true. If true, it marks the predicted values 
        for the x_data with an x on the line
    :param show_residuals: By default true. If true, is shows the residuals with
        red, dotted lines.
    :return: None
    """
    x_data = reg.get_x_data()
    y_data = reg.get_y_data()
    predicted = reg.get_predicted_values()

    plt.figure("Regression")
    plt.plot(x_data, y_data, "rx", markersize=10)
    if show_predictands:
        plt.plot(x_data, predicted, "bx", markersize=10)
    if show_residuals:
        [
            plt.plot((x_data[i], x_data[i]), (y_data[i], predicted[i]), "r--")
            for i in range(len(predicted))
        ]
    # division by 5 was arbitrarily chosen
    margin = (max(x_data) - min(x_data)) / 5
    linspace = np.linspace(min(x_data) - margin, max(x_data) + margin)
    plt.plot(linspace, [reg.f(xi) for xi in linspace], "k-")
    plt.show()
