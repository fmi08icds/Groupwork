"""The functions in this module serve as a quick sanity check to evaluate the
output of our regression classes visually with out the need of notebooks or 
visualization engines like dash"""
import matplotlib.pyplot as plt
import numpy as np


def plot_2d(reg, show_predictands=False, show_residuals=True):
    """
    Plot data with one dependent and one independent variable to show a 2D plot.
    :param reg: An instance of a regression from a class in regression.py
    :param show_predictands: By default true. If true, it marks the predicted values
        for the x_data with an x on the line
    :param show_residuals: By default true. If true, is shows the residuals with red,
        dotted lines.
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
        for i in enumerate(predicted):
            plt.plot((x_data[i], x_data[i]), (y_data[i], predicted[i]), "r--")
    # division by 5 was arbitrarily chosen
    margin = (max(x_data) - min(x_data)) / 5
    linspace = np.linspace(min(x_data) - margin, max(x_data) + margin)
    plt.plot(linspace, [reg.f(xi) for xi in linspace], "k-")
    # plt.show()


def plot_3d(reg):
    """
    Plot data with one dependent and one independent variable to show a 2D plot.
    :param reg: An instance of a regression from a class in regression.py
    :param show_predictands: By default true. If true, it marks the predicted values
        for the x_data with an x on the line
    :param show_residuals: By default true. If true, is shows the residuals with red,
        dotted lines.
    :return: None
    """
    y_data = reg.y_data

    plt.figure("Regression " + reg.name)
    ax = plt.subplot(projection="3d")
    ax.scatter(reg.get_x_column(0), reg.get_x_column(1), y_data, marker="x")

    # division by 5 was chosen by convention
    margin_x1 = max(reg.get_x_column(0)) - min(reg.get_x_column(0)) / 5
    margin_x2 = max(reg.get_x_column(0)) - min(reg.get_x_column(0)) / 5
    linspace_x1 = np.linspace(
        min(reg.get_x_column(0)) - margin_x1, max(reg.get_x_column(0)) + margin_x1
    )
    linspace_x2 = np.linspace(
        min(reg.get_x_column(1)) - margin_x2, max(reg.get_x_column(1)) + margin_x2
    )
    x_lin, y_lin = np.meshgrid(linspace_x1, linspace_x2)
    z = np.asarray(
        [
            [reg.f([x_lin[i, j], y_lin[i, j]]) for j in range(x_lin.shape[1])]
            for i in range(x_lin.shape[0])
        ]
    )
    ax.plot_surface(
        x_lin, y_lin, z, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3
    )


def plot_gaussians(reg, x):
    """
    Plots the gaussians

    :param reg: The regression model
    :param x: Ths point to evaluate
    """
    plt.figure("Gaussians " + reg.name)
    summed_gauss = sum(
        reg.gauss(reg.centres[index], x, reg.sigma) for index in range(len(reg.coeffs))
    )
    for ci in reg.centres:
        plt.plot(x, [reg.gauss(ci, xi, reg.sigma) / summed_gauss for xi in x])


def plot_residuals(reg):
    """
    Plots the residuals of the given regression model

    :param reg: The regression model
    """
    plt.figure("Residuals " + reg.name)
    # create residual presentation
    plt.plot(x=reg.y_data, y=reg.predicted_values, markersize=3)
