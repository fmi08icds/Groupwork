import numpy as np
import cv2 as cv
from enum import Enum


class conv_layer:
    '''
    '''
    def __init__(self, in_dim, conv_size=(3,3), kernel_num=4, debug = False, name = "conv"):
        self.kernel_num = kernel_num
        self.conv_size = conv_size
        self.conv_kernels = [None] * self.kernel_num
        self.name = name
        for i in range(0, self.kernel_num):
            if debug:
                self.conv_kernels[i] = self.debug_conv((conv_size[0], conv_size[1], in_dim[2]))
            else:
                self.conv_kernels[i] = np.random.uniform(-1,1,(conv_size[0], conv_size[1], in_dim[2]))

        self.in_dim = in_dim
        self.out_dim = (self.in_dim[0] - (conv_size[0] - 1), self.in_dim[1] - (self.conv_size[1] - 1), self.kernel_num)

        '''
    perform a forward convolution on the specified image.

    img -> np.array of shape (h,w,d)
    
    ret:
    out_img -> np.array of shape (h,w,d)
    '''
    def forward(self, img):
        out_img = np.zeros(self.out_dim)
        for k in range(0, self.kernel_num):
            for h in range(0, self.out_dim[0]):
                for w in range(0, self.out_dim[1]):
                    out_img[h, w, k] = np.sum(img[h:h+self.conv_size[0], w:w+self.conv_size[1],:] * self.conv_kernels[k])
        return out_img
    
    '''
    get out put dimension of this network layer

    ret:
    out_dim -> tuple of shape (h,w,d)
    '''
    def get_out_dim(self):
        return self.out_dim
    
    def debug_conv(self, size):
        kernel = np.zeros(size)
        for i in range(0, size[0],2):
            kernel[i,:,:] = 1
        return kernel
    
    def to_string(self):
        return self.name + " in_dim: " + str(self.in_dim) + " out_dim: " + str(self.out_dim)

class max_pooling_layer:
    '''
    Max pooling layer expects an input dimension (in_dim) of shape (h, w, d), where h and w
    are hight and width of images and d the number of dimensions. 
    
    Currently there is no padding or stride configuration. The layer operates with no paddling 
    and a stride of one.

    in_dim -> tuple of shape (h,w,d) -> input image dimensions 
    pooling_size -> tuple of shape (h, w) -> size of pooling filter
    '''
    def __init__(self, in_dim, pooling_size=(3,3), name = "max_pooling"):
        self.pooling_size = pooling_size
        self.in_dim = in_dim
        self.name = name
        h_overflow = 1 if self.in_dim[0] % self.pooling_size[0] > 0 else 0
        w_overflow = 1 if self.in_dim[1] % self.pooling_size[1] > 0 else 0
        self.out_dim = (int(self.in_dim[0] / self.pooling_size[0]) + h_overflow, int(self.in_dim[0] / self.pooling_size[0]) + w_overflow, self.in_dim[2])
        
    '''
    perform forward pooling on the specified image.

    img -> np.array of shape (h,w,d)
    
    ret:
    out_img -> np.array of shape (h,w,d)
    '''
    def forward(self, img):
        out_img = np.empty(self.out_dim)
        h_overflow = True if self.in_dim[0] / self.pooling_size[0] - self.out_dim[0] > 0 else False
        w_overflow = True if self.in_dim[1] / self.pooling_size[1] - self.out_dim[1] > 0 else False
        for d in range(0, self.out_dim[2]):
            for w in range(0, self.out_dim[0]):
                for h in range(0, self.out_dim[1]):
                    pool_size_h = self.pooling_size[0]
                    pool_size_w = self.pooling_size[1]
                    if h_overflow and h == (self.out_dim[0]-1):
                        pool_size_h = self.in_dim[0] % self.pooling_size[0]
                    if h_overflow and h == (self.out_dim[0]-1):
                        pool_size_w = self.in_dim[1] % self.pooling_size[1]

                    out_img[h,w,d] = np.max(img[h*pool_size_h:h*pool_size_h+pool_size_h, w*pool_size_w:w*pool_size_w+pool_size_w,d])
        return out_img
    '''
    get out put dimension of this network layer

    ret:
    out_dim -> tuple of shape (h,w,d)
    '''
    def get_out_dim(self):
        return self.out_dim
    
    def to_string(self):
        return self.name + " in_dim: " + str(self.in_dim) + " out_dim: " + str(self.out_dim)



class activation_layer:
    '''
    ReLU / sigmoid activation layer performs a activation function on each element in the input image. 

    in_dim -> tuple of shape (h,w,d) -> input image dimensions 
    activation_func -> string that is either 'relu' or 'sigmoid' -> switch type of activation 
    '''
    def __init__(self, in_dim, activation_func = 'relu', name = "activation"):
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.name = name
        self.possible_act_funcs = ['relu', 'sigmoid']
        if activation_func not in  self.possible_act_funcs:
            raise ValueError('activation layer only supports: ' + ', '.join(self.possible_act_funcs))
        self.activation_func = activation_func

    def forward(self, img):
        if self.activation_func == self.possible_act_funcs[0]:
            out_img = np.stack(np.vectorize(self.relu)(img), axis=0)
            return out_img
        else:
            out_img = np.stack(np.vectorize(self.sigmoid)(img), axis=0)
            return out_img
        

    def relu(self, el):
        return(np.maximum(0, el))
    
    def sigmoid(self, el):
        return(1/(1 + np.exp(-el)))
    
    def get_out_dim(self):
        return self.out_dim
    
    def to_string(self):
        return self.name + " " + self.activation_func +" in_dim: " + str(self.in_dim) + " out_dim: " + str(self.out_dim)
    
class fully_connected_layer:
    '''
    The fully connected layer transforms the input into a fully connected network
    with an output vector of out_dim.

    in_dim -> tuple of shape (h,w,d) -> input image dimensions 
    out_dim -> int -> defines the number of output nodes
    '''
    def __init__(self, in_dim, out_dim, name = "fully_connected"):
        self.in_dim = in_dim
        self.out_dim = (out_dim, 1)
        self.name = name
        w_dim = 1
        for d in in_dim:
            w_dim = w_dim * d
        self.weights = np.ones((self.out_dim[0], w_dim))
    
    
    def forward(self, img):
        out_vec = np.zeros(self.out_dim)
        img_vec = img.flatten()
        
        for i in range(0, self.out_dim[0]):
            out_vec[i][0] = np.sum(img_vec * self.weights[i])
        return out_vec

    def get_out_dim(self):
        return self.out_dim
    
    def to_string(self):
        return self.name + " in_dim: " + str(self.in_dim) + " out_dim: " + str(self.out_dim)

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


class test_layers_cnn:
    '''
    Orchestrates the forward and backward pass of the neural network.
    Includes the layers of the network: Convolutional layer, activation layer, max pooling layer, fully connected layer, softmax layer.
    '''

    def __init__(self, in_dim, out_dim):
        conv_1 = conv_layer(in_dim=in_dim, conv_size=(3,3), kernel_num=2)
        max_pool_1 = max_pooling_layer(in_dim=conv_1.get_out_dim(), pooling_size=(3,3))
        act_1 = activation_layer(in_dim=max_pool_1.get_out_dim(), activation_func='relu')
        fc_1 = fully_connected_layer(in_dim=act_1.get_out_dim(), out_dim=5)
        act_2 = activation_layer(in_dim=fc_1.get_out_dim(), activation_func='sigmoid')
        fc_2 = fully_connected_layer(in_dim=fc_1.get_out_dim(), out_dim=out_dim)
        act_3 = activation_layer(in_dim=fc_2.get_out_dim(), activation_func='sigmoid')

        self.layers = [
            conv_1, max_pool_1, act_1, fc_1, act_2, fc_2, act_3
        ]
    

    def forward(self, img):
        for layer in self.layers:
            img = layer.forward(img)
        return img

    def backward(self, out, learning_rate):
        for layer in reversed(self.layers):
            out = layer.backward(out, learning_rate)
        return out
    

class alex_net_cnn:
    """
    this network is inspired by AlexNet witch was developed by Alex Krizhevsky et al. in 2012 
    """
    def __init__(self) -> None:
        conv_1 = conv_layer(in_dim=(224,224,1), conv_size=(11,11), kernel_num=96)
        max_pool_1 = max_pooling_layer(in_dim=conv_1.get_out_dim(), pooling_size=(3,3))
        conv_2 = conv_layer(in_dim=max_pool_1.get_out_dim(), conv_size=(5,5), kernel_num=256)
        max_pool_2 = max_pooling_layer(in_dim=conv_2.get_out_dim(), pooling_size=(3,3))
        conv_3 = conv_layer(in_dim=max_pool_2.get_out_dim(), conv_size=(3,3), kernel_num=384)
        conv_4 = conv_layer(in_dim=conv_3.get_out_dim(), conv_size=(3,3), kernel_num=384)
        conv_5 = conv_layer(in_dim=conv_4.get_out_dim(), conv_size=(3,3), kernel_num=384)
        conv_6 = conv_layer(in_dim=conv_5.get_out_dim(), conv_size=(3,3), kernel_num=256)
        max_pool_3 = max_pooling_layer(in_dim=conv_6.get_out_dim(), pooling_size=(3,3))
        fc_1 = fully_connected_layer(in_dim=max_pool_3.get_out_dim(), out_dim=4095)
        fc_2 = fully_connected_layer(in_dim=fc_1.get_out_dim(), out_dim=4095)
        fc_3 = fully_connected_layer(in_dim=fc_2.get_out_dim(), out_dim=2)
        ac_1 = activation_layer(in_dim=fc_3.get_out_dim(), activation_func="sigmoid")

        self.layers = [
            conv_1, max_pool_1, conv_2, max_pool_2, conv_3, conv_4, conv_5, conv_6, max_pool_3, fc_1, fc_2, fc_3, ac_1
        ]

    def forward(self, img):
        for layer in self.layers:
            print(layer.to_string())
            img = layer.forward(img)
        return img

    def backward(self, out, learning_rate):
        for layer in reversed(self.layers):
            out = layer.backward(out, learning_rate)
        return out
# below add the resnet solution with pytorch
