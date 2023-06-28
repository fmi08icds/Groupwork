import math

import numpy as np

class LocallyWeightedRegression:
    coeffs = None
    predicted_values = None
    x_data = None
    y_data = None
    # W = None

    def __init__(self, data, transposed=False, tau=.5, sections=None):
        """
        Calculates the ordinary linear regression for the given data.
        :param data: data is a Nx(d+1) matrix with the last column being Y and X being Nxd. data[0] accesses therefor the first sample.
        :param transposed: If the data is transposed and data[0] returns a vector of the first dimension of the samples.
        """
        data = np.asarray(data)
        if transposed:
            data = np.transpose(data)

        self.y_data = data[:, -1]
        # removes y-column and adds column of ones for the bias (w0)
        # self.x_data = np.asarray([np.insert(sample[:-1], 0, 1) for sample in data])
        self.x_data = data[:,:-1]
        x_data = np.asarray([np.insert(sample, 0, 1) for sample in self.x_data])
        # raise Exception
        def w(i):
            return [math.e ** (-(np.linalg.norm(x_data[i] - x_j)) ** 2 / (2 * tau ** 2)) for x_j in x_data]
        if sections is None:
            W = np.zeros((len(x_data), len(x_data), len(x_data)))
            for i in range(len(W)):
               W[i] = np.diag(w(i))
            self.centres = x_data[:,1]
        else:
            x_sorted = x_data[x_data[:,1].argsort()]
            x_1_sorted = x_data[:,1].argsort()
            W = np.zeros((sections, len(x_data), len(x_data)))
            sec_len = len(x_sorted)//sections
            prev_indices = 0
            self.centres = []
            for i in range(sections):
                sec = x_sorted[i*sec_len: (1+i)*sec_len][:,1]
                self.centres.append(np.mean(sec))
                index = len(sec)//2 + prev_indices
                W[i] = np.diag(w(index))
                prev_indices += len(sec)
            self.centres = np.asarray(self.centres)

        def get_coeffs_i(i):
            return np.linalg.pinv(np.transpose(x_data) @ W[i] @ x_data) @ np.transpose(x_data) @ W[
                i] @ self.y_data

        self.coeffs = [get_coeffs_i(i) for i in range(len(W))]
        self.predicted_values = np.asarray([self.f(xi) for xi in x_data[:, 1:]])

    def gauss(self, centre, x, sigma=1): return math.e ** (-(centre - x) ** 2 / (2 * sigma ** 2))

    def f(self, x):
        if type(x) is not np.array(()):
            if type(x) in [int, float]:
                x = [x]
            x = np.asarray([x])
        if len(x) != len(self.coeffs[0]) - 1:
            raise ValueError(
                f"x has to have the same dimension as x_data. dim x: {len(x)}; dim x_data: {len(self.x_data[0])}")


        summed = 0
        summed_gauss = sum([self.gauss(self.centres[index], x) for index in range(len(self.coeffs))])
        for index, coeff in enumerate(self.coeffs):
            summed += self.gauss(self.centres[index], x)/ summed_gauss * sum(coeff * np.insert(x, 0, 1))
        while True:
            try:
                summed = summed[0]
            except IndexError:
                return summed


    def get_sum_of_squares(self):
        return np.sum([(self.predicted_values[i] - self.y_data[i]) ** 2 for i in range(len(self.y_data))])
