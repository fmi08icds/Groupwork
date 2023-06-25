import numpy as np


class linear_regression:
    weights = None
    predicted_values = None
    x_data = None
    y_data = None
    
    def __init__(self, data, transposed=False):
        """
        Calculates the ordinary linear regression for the given data.
        :param data: data is a Nx(d+1) matrix with the last column being Y and X being Nxd. data[0] accesses therefor the first sample.
        :param transposed: If the data is transposed and data[0] returns a vector of the first dimension of the samples.
        """
        global weights
        global predicted_values
        global x_data
        global y_data
        data = np.asarray(data)
        if transposed:
            data = np.transpose(data)
            
        y_data = data[:,-1]
        # removes y-column and ads column of ones for the bias (w0)
        x_data = np.asarray([np.insert(sample[:-1], 0, 1) for sample in data])

        weights = np.linalg.pinv(np.transpose(x_data) @ x_data) @ np.transpose(x_data) @ y_data
        predicted_values = x_data @ weights

    def get_weights(self):
        # global weights
        return weights



    def get_predicted_values(self):
        # global predicted_values
        return predicted_values


    def get_sum_of_squares(self):
        return np.sum([(predicted_values[i] - y_data[i])**2 for i in range(len(y_data))])


    def get_x_data(self):
        return x_data[:, 1:]


    def get_y_data(self):
        return y_data
