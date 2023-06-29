import matplotlib.pyplot as plt
import numpy as np


def plot_2d(reg, show_predictands=False, show_residuals=True):
    """
    Plot data with one dependent and one independent variable to show a 2D plot.
    :param reg: An instance of a regression from a class in regression.py
    :param show_predictands: By default true. If true, it marks the predicted values for the x_data with an x on the line
    :param show_residuals: By default true. If true, is shows the residuals with red, dotted lines.
    :return: None
    """
    x_data = reg.x_data
    y_data = reg.y_data
    predicted = reg.predicted_values

    plt.figure("Regression " + reg.name)
    plt.plot(x_data, y_data, "rx", markersize=3)
    if show_predictands:
        plt.plot(x_data, predicted, "bx", markersize=3)
    if show_residuals:
        [plt.plot((x_data[i], x_data[i]), (y_data[i], predicted[i]), "r--") for i in range(len(predicted))]
    # division by 5 was arbitrarily chosen
    margin = (max(x_data) - min(x_data))/5
    linspace = np.linspace(min(x_data)-margin, max(x_data)+margin)
    plt.plot(linspace, [reg.f(xi) for xi in linspace], "k-")
    # plt.show()


def plot_gaussians(reg, x):
    plt.figure("Gaussians " + reg.name)
    summed_gauss = sum([reg.gauss(reg.centres[index], x, reg.sigma) for index in range(len(reg.coeffs))])
    [plt.plot(x, [reg.gauss(ci, xi, reg.sigma)/ summed_gauss for xi in x]) for ci in reg.centres]