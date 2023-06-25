import types

import matplotlib.pyplot as plt
import numpy as np
import importlib.machinery
import importlib.util
import os, sys

# Imported according to https://stackoverflow.com/questions/19009932/import-arbitrary-python-source-file-python-3-3
# loader = importlib.machinery.SourceFileLoader("regression", os.path.join(os.path.dirname(__file__), '../..', 'regression.py'))
spec = importlib.util.spec_from_file_location("regression", os.path.join(os.path.dirname(__file__), '../..', 'regression.py'))
module = importlib.util.module_from_spec(spec)
sys.modules["regression"] = module
spec.loader.exec_module(module)


def plot_2d(reg, show_predictands=True, show_residuals=True):
    """
    Plot data with one dependent and one independent variable to show a 2D plot.
    :param reg: An instance of a regression from a class in regression.py
    :param show_predictands: By default true. If true, it marks the predicted values for the x_data with an x on the line
    :param show_residuals: By default true. If true, is shows the residuals with red, dotted lines.
    :return: None
    """
    x_data = reg.get_x_data()
    y_data = reg.get_y_data()
    predicted = reg.get_predicted_values()
    coeffs = reg.get_coeffs()
    def f(x): return [coeffs[0] + coeffs[1] * x_i for x_i in x]

    plt.figure("Regression")
    plt.plot(x_data, y_data, "rx", markersize=10)
    if show_predictands:
        plt.plot(x_data, predicted, "bx", markersize=10)
    if show_residuals:
        [plt.plot((x_data[i], x_data[i]), (y_data[i], predicted[i]), "r--") for i in range(len(predicted))]
    # division by 5 was arbitrarily chosen
    margin = (max(x_data) - min(x_data))/5
    linspace = np.linspace(min(x_data)-margin, max(x_data)+margin)
    plt.plot(linspace, f(linspace), "k-")
    plt.show()
