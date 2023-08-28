import numpy as np
import cv2 as cv
from enum import Enum
from tqdm import tqdm

import preparation
import util

import torch
from torch.utils.data import random_split
from torch import cuda, device
from torch import nn
from torch import optim, from_numpy, tensor
## COMMENTS: You mentioned in your requirement file we only need numpy to run your main. What about all other external libraries imported above??

class conv_layer: ## COMMENTS: as suggested in class better to name your class with capital letters to distinguish them from functions
    '''
    Convolution layer expects an input dimension (in_dim) of shape (h, w, d), where h and w
    are hight and width of images and d the number of dimensions.

    Currently there is no padding or stride configuration. The layer operates with no paddling
    and a stride of one.

    in_dim -> tuple of shape (h,w,d) -> input image dimensions
    conv_size -> tuple of shape (h, w) -> size of convolution kernel
    kernel_num -> int -> number of kernels
    '''

    def __init__(self, in_dim, conv_size, kernel_num, debug=False):
        self.kernel_num = kernel_num
        self.conv_size = conv_size
        self.conv_kernels = [None] * self.kernel_num

        for i in range(0, self.kernel_num):
            if debug:
                self.conv_kernels[i] = self.debug_conv(
                    (conv_size[0], conv_size[1], in_dim[2]))
            else:
                self.conv_kernels[i] = np.random.uniform(
                    -1, 1, (conv_size[0], conv_size[1], in_dim[2]))

        self.in_dim = in_dim
        self.out_dim = (self.in_dim[0] - (conv_size[0] - 1),
                        self.in_dim[1] - (self.conv_size[1] - 1), self.kernel_num)
    '''
    perform a forward convolution on the specified image.

    img -> np.array of shape (h,w,d)

    ret:
    out_img -> np.array of shape (h,w,d)
    '''

    def forward(self, img):
        self.input = img ## COMMENTS: Please initialise this attribute in the constructor.
        out_img = np.zeros(self.out_dim)
        for k in range(0, self.kernel_num):
            for h in range(0, self.out_dim[0]):
                for w in range(0, self.out_dim[1]):
                    out_img[h, w, k] = np.sum(
                        img[h:h+self.conv_size[0], w:w+self.conv_size[1], :] * self.conv_kernels[k])
        return out_img

    def backward(self, grad_output, learning_rate):
        grad_input = np.zeros(self.in_dim)
        for k in range(self.kernel_num):
            for h in range(self.out_dim[0]):
                for w in range(self.out_dim[1]):
                    grad_input[h:h+self.conv_size[0], w:w+self.conv_size[1],
                               :] += grad_output[h, w, k] * self.conv_kernels[k]
                    self.conv_kernels[k] -= learning_rate * grad_output[h, w, k] *
                        self.input[h:h+self.conv_size[0],
                                   w:w+self.conv_size[1], :]
        return grad_input

    '''
    get out put dimension of this network layer

    ret:
    out_dim -> tuple of shape (h,w,d)
    '''

    def get_out_dim(self):
        return self.out_dim

    def debug_conv(self, size):
        kernel = np.zeros(size)
        for i in range(0, size[0], 2):
            kernel[i, :, :] = 1
        return kernel


class max_pooling_layer:
    '''
    Max pooling layer expects an input dimension (in_dim) of shape (h, w, d), where h and w
    are hight and width of images and d the number of dimensions.

    Currently there is no padding or stride configuration. The layer operates with no paddling
    and a stride of one.

    in_dim -> tuple of shape (h,w,d) -> input image dimensions
    pooling_size -> tuple of shape (h, w) -> size of pooling filter
    '''

    def __init__(self, in_dim, pooling_size):
        self.pooling_size = pooling_size
        self.in_dim = in_dim
        h_overflow = 1 if self.in_dim[0] % self.pooling_size[0] > 0 else 0
        w_overflow = 1 if self.in_dim[1] % self.pooling_size[1] > 0 else 0
        self.out_dim = (int(self.in_dim[0] / self.pooling_size[0]) + h_overflow, int(
            self.in_dim[0] / self.pooling_size[0]) + w_overflow, self.in_dim[2])

    '''
    perform forward pooling on the specified image.

    img -> np.array of shape (h,w,d)

    ret:
    out_img -> np.array of shape (h,w,d)
    '''

    def forward(self, img):
        self.input = img
        out_img = np.empty(self.out_dim)
        h_overflow = True if self.in_dim[0] / \
            self.pooling_size[0] - self.out_dim[0] > 0 else False
        w_overflow = True if self.in_dim[1] / \
            self.pooling_size[1] - self.out_dim[1] > 0 else False
        for d in range(0, self.out_dim[2]):
            for w in range(0, self.out_dim[0]):
                for h in range(0, self.out_dim[1]):
                    pool_size_h = self.pooling_size[0]
                    pool_size_w = self.pooling_size[1]
                    if h_overflow and h == (self.out_dim[0]-1):
                        pool_size_h = self.in_dim[0] % self.pooling_size[0]
                    if w_overflow and w == (self.out_dim[0]-1):
                        pool_size_w = self.in_dim[1] % self.pooling_size[1]

                    out_img[h, w, d] = np.max(
                        img[h*pool_size_h:h*pool_size_h+pool_size_h, w*pool_size_w:w*pool_size_w+pool_size_w, d])
        return out_img

    def backward(self, grad_output, learning_rate):

        grad_output = np.reshape(grad_output, (self.out_dim))
        grad_input = np.zeros(self.in_dim)

        h_overflow = True if self.in_dim[0] / \
            self.pooling_size[0] - self.out_dim[0] > 0 else False
        w_overflow = True if self.in_dim[1] / \
            self.pooling_size[1] - self.out_dim[1] > 0 else False

        count_grad_slice = 0
        for d in range(self.out_dim[2]):
            for w in range(self.out_dim[0]):
                for h in range(self.out_dim[1]):
                    pool_size_h = self.pooling_size[0]
                    pool_size_w = self.pooling_size[1]
                    if h_overflow and h == (self.out_dim[0]-1):
                        pool_size_h = self.in_dim[0] % self.pooling_size[0]
                    if w_overflow and w == (self.out_dim[1]-1):
                        pool_size_w = self.in_dim[1] % self.pooling_size[1]

                    grad_slice = grad_output[w, h, d]
                    count_grad_slice += 1
                    mask = (self.input[w*pool_size_h:w*pool_size_h+pool_size_h, h*pool_size_w:h*pool_size_w+pool_size_w, d] == np.max(
                        self.input[w*pool_size_h:w*pool_size_h+pool_size_h, h*pool_size_w:h*pool_size_w+pool_size_w, d]))
                    grad_input[w*pool_size_h:w*pool_size_h+pool_size_h, h*pool_size_w:h *
                               pool_size_w+pool_size_w, d] = mask * grad_slice  # !!! h und w vertauscht?
                    # grad_input[h * self.pooling_size[0]:h * self.pooling_size[0] + pool_size_h, w * self.pooling_size[1]:w * self.pooling_size[1] + pool_size_w, d] += grad_output[h, w, d] * mask
        return grad_input

    '''
    get out put dimension of this network layer

    ret:
    out_dim -> tuple of shape (h,w,d)
    '''

    def get_out_dim(self):
        return self.out_dim


class fully_connected_layer:
    '''
    The fully connected layer transforms the input into a fully connected network
    with an output vector of out_dim.

    in_dim -> tuple of shape (h,w,d) -> input image dimensions
    out_dim -> int -> defines the number of output nodes
    '''

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        w_dim = 1
        for d in in_dim:
            w_dim = w_dim * d
        self.weights = np.random.randn(self.out_dim, w_dim)

    def forward(self, img):
        self.img = img
        out_vec = np.zeros(self.out_dim)
        img_vec = img.flatten()

        for i in range(0, self.out_dim):
            out_vec[i] = np.sum(img_vec * self.weights[i])
        return out_vec

    def backward(self, grad_output, learning_rate):

        grad_output = np.reshape(grad_output, (np.prod(grad_output.shape), 1))
        self.img = np.reshape(self.img, (np.prod(self.img.shape), 1))

        grad_weights = np.dot(grad_output, self.img.T)
        grad_input = np.dot(self.weights.T, grad_output)
        self.weights -= learning_rate * grad_weights
        return grad_input

    def get_out_dim(self):
        return self.out_dim


class sigmoid_activation_layer():
    '''
    Sigmoid function and its derivative for the activation of the values of each layer
    '''

    def __init__(self):
        pass

    def sigm(self, x):
        sigm_res = 1 / (1 + np.exp(-x))
        return sigm_res

    def sigm_deriv(self, x):
        sigm_deriv_res = self.sigm(x) * (1 - self.sigm(x))
        return sigm_deriv_res

    def forward(self, img):
        self.input = img
        self.fw_res = self.sigm(self.input)
        return self.fw_res

    def backward(self, grad_output, learning_rate): ##COMMENTS: the two input parameters are not used in the methods.
        self.bw_res = self.sigm_deriv(self.input)
        return self.bw_res


def bin_cross_entropy(y_true, y_pred):
    '''
    Binary cross entropy loss function describes the loss between the true and the predicted value
    '''
    bin_cross_entropy_res = - \
        np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
    return bin_cross_entropy_res


def bin_cross_entropy_deriv(y_true, y_pred):
    '''
    Derivative of the binary cross entropy loss function
    '''
    bin_cross_entropy_deriv_res = (
        1/np.size(y_true))*(((1-y_true)/(1-y_pred))-(y_true/y_pred))
    return bin_cross_entropy_deriv_res


def pred(cnn, img):
    '''
    Run forward pass of the network to make predictions
    '''
    fw = img
    for layer in cnn:
        fw = layer.forward(fw)
    return fw


def train(cnn, x_train, y_train, x_val, y_val, epochs, learning_rate):
    '''
    Train a given network with data from x_train and y_train
    '''
    for epoch in range(0, epochs):
        loss = 0
        t_bar = tqdm(enumerate(zip(x_train, y_train)),
                     total=len(x_train))  # !!!
        for index, (img, img_label) in t_bar:

            fw = pred(cnn, img)
            fw = np.reshape(fw, (2, 1))
            loss += bin_cross_entropy(img_label, fw)
            # t_bar.set_description('Training loss: %s'%(round(loss, 4)))

            grad = bin_cross_entropy_deriv(img_label, fw)
            for layer in reversed(cnn):
                grad = layer.backward(grad, learning_rate)

            # t_bar.set_description('Training loss: %s'%(round(loss_end, 4)))
            t_bar.set_description('Training loss: %s' % (round(loss, 4)))
        loss_end = loss/(index+1)
        print('Epoch %s/%s | Training loss: %s' %
              (epoch, epochs, round(loss_end, 4)))
        print('Validation:')
        test(cnn, x_val, y_val)


def test(cnn, x_test, y_test):
    '''
    Test a given network with data from x_test and y_test
    '''
    confusion_matrix = util.confusion_matrix(cnn, x_test, y_test)

    accuracy = util.accuracy(confusion_matrix)
    print('Accuracy:', accuracy)
    precision = util.precision(confusion_matrix)
    print('Precision:', precision)
    recall = util.recall(confusion_matrix)
    print('Recall:', recall)


def run_base_cnn(split_data, classes_data, epochs, learning_rate):
    '''
    Run the created naive base cnn with the given data
    '''
    kernel_num = 3
    # base_cnn = [
    #     conv_layer(in_dim=preparation.reshape_img(
    #         split_data[0][0]).shape, conv_size=(3, 3), kernel_num=kernel_num),
    #     sigmoid_activation_layer(),
    #     max_pooling_layer(in_dim=(98, 98, kernel_num), pooling_size=(4, 4)),
    #     conv_layer(in_dim=(25, 25, kernel_num), conv_size=(3, 3), kernel_num=kernel_num),
    #     sigmoid_activation_layer(),
    #     max_pooling_layer(in_dim=(23, 23, kernel_num), pooling_size=(2, 2)),
    #     fully_connected_layer(in_dim=(12, 12, kernel_num), out_dim=100),
    #     sigmoid_activation_layer(),
    #     fully_connected_layer(in_dim=(100, 1), out_dim=20),
    #     sigmoid_activation_layer(),
    #     fully_connected_layer(in_dim=(20, 1), out_dim=2),
    #     sigmoid_activation_layer()
    # ]
    base_cnn = [
        conv_layer(in_dim=preparation.reshape_img(
            split_data[0][0]).shape, conv_size=(3, 3), kernel_num=kernel_num),
        max_pooling_layer(in_dim=(98, 98, kernel_num), pooling_size=(3, 3)),
        sigmoid_activation_layer(),
        fully_connected_layer(in_dim=(33, 33, kernel_num), out_dim=200),
        sigmoid_activation_layer(),
        fully_connected_layer(in_dim=(200, 1), out_dim=20),
        sigmoid_activation_layer(),
        fully_connected_layer(in_dim=(20, 1), out_dim=2),
        sigmoid_activation_layer()
    ]

    train(base_cnn, split_data[0], classes_data[0], split_data[1], classes_data[1],
          epochs=epochs, learning_rate=learning_rate)

    test(base_cnn, split_data[2], classes_data[2])


class TorchCNN(nn.Module):
    '''
    '''

    def __init__(self):
        super(TorchCNN, self).__init__()
        self.c1 = nn.Conv2d(1, 96, 11, stride=4)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3,stride=2)
        self.c2 = nn.Conv2d(96, 256, 3, padding=2)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3,stride=2)
        self.f1 = nn.Linear(50176, 4096)
        self.act3 = nn.ReLU()
        self.f2 = nn.Linear(4096, 128)
        self.act4 = nn.ReLU()
        self.f3 = nn.Linear(128, 2)

    def forward(self, fw):
        fw = self.c1(fw)
        fw = self.act1(fw)
        fw = self.pool1(fw)
        fw = self.c2(fw)
        fw = self.act2(fw)
        fw = self.pool2(fw)
        fw = fw.view(fw.size(0), -1)
        fw = self.f1(fw)
        fw = self.act3(fw)
        fw = self.f2(fw)
        fw = self.act4(fw)
        fw = self.f3(fw)
        return fw


def train_torch_cnn(base_torch_model, train_load, val_load, epochs, loss_func, optimizer, device):
    for epoch in range(epochs):
        base_torch_model.train()
        loss_run = 0.0
        for images, labels in train_load:
            images, labels = images.to(device), labels.squeeze().to(device)
            optimizer.zero_grad()
            out = base_torch_model(images)
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()
            loss_run += loss.item()
            loss = 0
        loss_res = loss_run / len(train_load)
        print('Epoch:', epoch + 1, '/', epochs, 'Loss:', loss_res)
        base_torch_model.eval()
        true, all = 0, 0
        with torch.no_grad():
            print('start validation...')
            for images, labels in val_load:
                images, labels = images.to(device), labels.squeeze().to(device)
                out = base_torch_model(images)
                _, predicted = torch.max(out.data, 1)
                _, labels_max = torch.max(labels, 1)
                all += labels.size(0)
                true += (predicted == labels_max).sum().item()

        acc = true / all
        print('Validation acc', acc)


def test_torch_cnn(base_torch_model, test_load, loss_func, device):
    base_torch_model.eval()
    true, all, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        print('start testing...')
        for images, labels in test_load:
            images, labels = images.to(device), labels.squeeze().to(device)
            out = base_torch_model(images)
            _, predicted = torch.max(out.data, 1)
            _, labels_max = torch.max(labels, 1)
            all += labels.size(0)
            true += (predicted == labels_max).sum().item()
            for pred, label in zip(predicted, labels_max):
                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 0 and label == 0:
                    tn += 1
                elif pred == 0 and label == 1:
                    fn += 1
                elif pred == 1 and label == 0:
                    fp += 1
    acc = true / all
    print('\nTP %s, TN %s, FN %s, FP %s' % (tp, tn, fn, fp))
    print('Accuracy:', acc)
    print('Precision:', float(tp / (tp + fp)))
    print('Recall:', float(tp / (tp + fn)))


def run_torch_cnn(split_data, classes_data, epochs, learning_rate, batch_size):
    '''
    Run the cnn model from pytorch with the given data
    '''
    train_load, val_load, test_load = preparation.torch_cnn_prepare_data(
        split_data, classes_data, batch_size)
    base_torch_model = TorchCNN()
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(base_torch_model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if cuda.is_available() else "cpu")
    base_torch_model.to(device)

    train_torch_cnn(base_torch_model, train_load,
                    val_load, epochs, loss_func, optimizer, device)
    test_torch_cnn(base_torch_model, test_load, loss_func, device)
    torch.save(base_torch_model.state_dict(),"models/base_torch_cnn")


# class test_layers_cnn:
#     '''
#     Orchestrates the forward and backward pass of the neural network.
#     Includes the layers of the network: Convolutional layer, activation layer, max pooling layer, fully connected layer, softmax layer.
#     '''

#     def __init__(self, in_dim, out_dim):
#         conv_1 = conv_layer(in_dim=in_dim, conv_size=(3, 3), kernel_num=2)
#         max_pool_1 = max_pooling_layer(
#             in_dim=conv_1.get_out_dim(), pooling_size=(3, 3))
#         act_1 = activation_layer(
#             in_dim=max_pool_1.get_out_dim(), activation_func='relu')
#         fc_1 = fully_connected_layer(in_dim=act_1.get_out_dim(), out_dim=5)
#         act_2 = activation_layer(
#             in_dim=fc_1.get_out_dim(), activation_func='sigmoid')
#         fc_2 = fully_connected_layer(
#             in_dim=fc_1.get_out_dim(), out_dim=out_dim)
#         act_3 = activation_layer(
#             in_dim=fc_2.get_out_dim(), activation_func='sigmoid')

#         self.layers = [
#             conv_1, max_pool_1, act_1, fc_1, act_2, fc_2, act_3
#         ]

#     def forward(self, img):
#         for layer in self.layers:
#             img = layer.forward(img)
#         return img

#     def backward(self, out, learning_rate):
#         for layer in reversed(self.layers):
#             out = layer.backward(out, learning_rate)
#         return out


# class alex_net_cnn:
#     """
#     this network is inspired by AlexNet witch was developed by Alex Krizhevsky et al. in 2012
#     """

#     def __init__(self) -> None:
#         conv_1 = conv_layer(in_dim=(224, 224, 1),
#                             conv_size=(11, 11), kernel_num=96)
#         max_pool_1 = max_pooling_layer(
#             in_dim=conv_1.get_out_dim(), pooling_size=(3, 3))
#         conv_2 = conv_layer(in_dim=max_pool_1.get_out_dim(),
#                             conv_size=(5, 5), kernel_num=256)
#         max_pool_2 = max_pooling_layer(
#             in_dim=conv_2.get_out_dim(), pooling_size=(3, 3))
#         conv_3 = conv_layer(in_dim=max_pool_2.get_out_dim(),
#                             conv_size=(3, 3), kernel_num=384)
#         conv_4 = conv_layer(in_dim=conv_3.get_out_dim(),
#                             conv_size=(3, 3), kernel_num=384)
#         conv_5 = conv_layer(in_dim=conv_4.get_out_dim(),
#                             conv_size=(3, 3), kernel_num=384)
#         conv_6 = conv_layer(in_dim=conv_5.get_out_dim(),
#                             conv_size=(3, 3), kernel_num=256)
#         max_pool_3 = max_pooling_layer(
#             in_dim=conv_6.get_out_dim(), pooling_size=(3, 3))
#         fc_1 = fully_connected_layer(
#             in_dim=max_pool_3.get_out_dim(), out_dim=4095)
#         fc_2 = fully_connected_layer(in_dim=fc_1.get_out_dim(), out_dim=4095)
#         fc_3 = fully_connected_layer(in_dim=fc_2.get_out_dim(), out_dim=2)
#         ac_1 = activation_layer(in_dim=fc_3.get_out_dim(),
#                                 activation_func="sigmoid")

#         self.layers = [
#             conv_1, max_pool_1, conv_2, max_pool_2, conv_3, conv_4, conv_5, conv_6, max_pool_3, fc_1, fc_2, fc_3, ac_1
#         ]

#     def forward(self, img):
#         for layer in self.layers:
#             img = layer.forward(img)
#         return img

#     def backward(self, out, learning_rate):
#         for layer in reversed(self.layers):
#             out = layer.backward(out, learning_rate)
#         return out
# # below add the resnet solution with pytorch