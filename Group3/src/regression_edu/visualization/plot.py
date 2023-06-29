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

def plot_3d(reg, show_predictands=False, show_residuals=True):
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
    ax = plt.subplot(projection="3d")
    ax.scatter(reg.get_x_column(0), reg.get_x_column(1), y_data, marker="x")
    # if show_predictands:
    #     plt.plot(x_data, predicted, "bx", markersize=3)
    # if show_residuals:
    #     [plt.plot((x_data[i], x_data[i]), (y_data[i], predicted[i]), "r--") for i in range(len(predicted))]
    # division by 5 was arbitrarily chosen
    margin_x1 = max(reg.get_x_column(0)) - min(reg.get_x_column(0))/5
    margin_x2 = max(reg.get_x_column(0)) - min(reg.get_x_column(0))/5
    linspace_x1 = np.linspace(min(reg.get_x_column(0))-margin_x1, max(reg.get_x_column(0))+margin_x1)
    linspace_x2 = np.linspace(min(reg.get_x_column(1))-margin_x2, max(reg.get_x_column(1))+margin_x2)
    x_lin, y_lin = np.meshgrid(linspace_x1, linspace_x2)
    # plt.plot(x_lin, y_lin, [[reg.f([xi,yi]) for yi in y_lin] for xi in x_lin])
    print(x_lin.shape[0])
    z = np.asarray([[reg.f([x_lin[i,j],y_lin[i,j]]) for j in range(x_lin.shape[1])] for i in range(x_lin.shape[0])])
    print(z.shape)
    # cont = ax.contour(x_lin, y_lin, z, 20)
    # plt.colorbar(cont, shrink=.5, aspect=10)
    ax.plot_surface(x_lin, y_lin, z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)
    # plt.show()


def plot_gaussians(reg, x):
    plt.figure("Gaussians " + reg.name)
    summed_gauss = sum([reg.gauss(reg.centres[index], x, reg.sigma) for index in range(len(reg.coeffs))])
    [plt.plot(x, [reg.gauss(ci, xi, reg.sigma)/ summed_gauss for xi in x]) for ci in reg.centres]