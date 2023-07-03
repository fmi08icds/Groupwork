import numpy as np
import cv2 as cv
import torch


class conv_layer:
    '''
    '''

    def __init__(self):
        pass


class max_pooling_layer:
    '''
    '''

    def __init__(self):
        pass


class activation_layer:
    '''
    '''

    def __init__(self):
        pass


class fully_connected_layer:
    '''
    '''

    def __init__(self):
        pass


class softmax_layer:
    '''
    forward pass describes normalization of the previous values and a formula of a softmax layer.
    backward pass describes the adaption of the previous output vectors by learning rate and gradient of the softmax layer and the restoration of the original values from normalization.

    in_dim -> all nodes of the output vector from the fully_connected_layer (default: 10)
    out_dim -> all nodes from softmax_layer (default: 1)
    '''

    def __init__(self, in_dim):
        self.in_dim = in_dim
        self.out_dim = 1

    def forward(self, img):
        self.last_img = img

        # img_norm = img / np.linalg.norm(img)
        img_norm = (img - np.mean(img)) / np.std(img)
        self.last_img_norm = img_norm

        out_probability = np.exp(
            img_norm)[0] / np.sum(np.exp(img_norm), axis=0)

        return out_probability

    # backward pass
    def backward(self, grad_output, learning_rate):
        softmax_output = self.forward(self.last_img)
        grad_input = softmax_output * (1 - softmax_output) * grad_output

        act_img = self.last_img - learning_rate * grad_input
        # act_img_restored = act_img * np.linalg.norm(self.last_img)
        act_img_restored = (act_img * np.std(self.last_img)
                            ) + np.mean(self.last_img)

        return act_img_restored


class naive_cnn:
    '''
    Orchestrates the forward and backward pass of the neural network.
    Includes the layers of the network: Convolutional layer, activation layer, max pooling layer, fully connected layer, softmax layer.
    '''

    def __init__(self):
        pass
        # self.layers = [
        #     conv_layer(in_dim=img.shape, conv_size=(3,3), kernel_num=2),
        #     activation_layer(in_dim=conv_layer.get_out_dim()),
        #     max_pooling_layer(in_dim=conv_layer.get_out_dim(), pooling_size=(3,3)),
        #     fully_connected_layer(in_dim=max_pooling_layer.get_out_dim(), out_dim=10),
        #     softmax_layer(in_dim=fully_connected_layer.get_out_dim())
        # ]

    def forward(self, img):
        for layer in self.layers:
            img = layer.forward(img)
        return img

    def backward(self, out, learning_rate):
        for layer in reversed(self.layers):
            out = layer.backward(out, learning_rate)
        return out

# below add the resnet solution with pytorch
