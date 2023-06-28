import os
import argparse
import shutil
import random
import numpy as np
import cv2 as cv
import torch
import matplotlib.pyplot as plt

import preparation
import model
import util

data_files_path = 'data/chest_xray/'
data_files_path_origin = 'data/chest_xray_origin/'
data_path = 'data'
data_files_train_split_path = ['train', 'test', 'val']

ratio_train = 0.7
ratio_test = 0.2
ratio_val = 0.1

ratio_test_val = [ratio_test, ratio_val, ratio_test, ratio_val]
source_train_folder = ['train/NORMAL', 'train/NORMAL',
                       'train/PNEUMONIA', 'train/PNEUMONIA']
target_test_val_folder = ['test/NORMAL',
                          'val/NORMAL', 'test/PNEUMONIA', 'val/PNEUMONIA']

data_directory = 'Group1/data/'
split = ['train', 'val', 'test']
classes = ['NORMAL', 'PNEUMONIA']
img_size = 224

learning_rate = 0.01

'''
Organize the variables for all classes
'''


class main_variables():
    def __init__(self, model_name):
        self.model_name = model_name
        self.data_files_path = 'data/chest_xray/'
        self.data_files_path_origin = 'data/chest_xray_origin/'
        self.data_path = 'data'
        self.data_files_train_split_path = ['train', 'test', 'val']
        self.ratio_train = 0.7
        self.ratio_test = 0.2
        self.ratio_val = 0.1
        self.ratio_test_val = [self.ratio_test,
                               self.ratio_val, self.ratio_test, ratio_val]
        self.source_train_folder = ['train/NORMAL', 'train/NORMAL',
                                    'train/PNEUMONIA', 'train/PNEUMONIA']
        self.target_test_val_folder = ['test/NORMAL',
                                       'val/NORMAL', 'test/PNEUMONIA', 'val/PNEUMONIA']
        self.data_directory = 'data/'
        self.split = ['train', 'val', 'test']
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.img_size = 224
        self.learning_rate = 0.01


def orchestrate_preperation(model_name):
    '''
    Orchestrate the preparation methods of the data
    '''
    args = main_variables(model_name)

    preparation.data_folder_init(
        args.data_path, args.data_files_path, args.data_files_path_origin)
    preparation.data_folder_train_split(
        args.data_path, args.ratio_test_val, args.source_train_folder, args.target_test_val_folder)


def exec_naive_model(split_data, classes_data):
    '''
    Execute the naive cnn model from model.py
    '''

    print('Executing naive model...')
    args = main_variables('naive')


def exec_resnet_model(split_data, classes_data):
    '''
    Execute the resnet model from model.py
    '''

    print('Executing resnet model...')
    args = main_variables('resnet')


def exec_inspection(split_data, classes_data):
    '''
    image selection and show for testing purposes from util.py
    '''

    print('Executing image inspection...')
    args = main_variables('resnet')

    img, img_label, rand_img_num = util.select_random_image(
        split_data, classes_data, split_set=0)
    util.show_random_image(args.classes, classes_data,
                           img, rand_img_num, split_set=0)


def orchestrate_model(model_name):
    '''
    Orchestrate the model execution
    '''
    args = main_variables(model_name)

    split_data, classes_data = preparation.read_training_data(
        args.data_directory, args.split, args.classes, args.img_size)

    if model_name == 'naive':  # run the naive cnn model
        exec_naive_model(split_data, classes_data)

    elif model_name == 'resnet':  # run the resnet model
        exec_resnet_model(split_data, classes_data)

    elif model_name == 'inspect':  # run a sample image
        exec_inspection(split_data, classes_data)

    print('Finished %s model.' % (model_name))


def parseArguments():
    '''
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', help='model to use',
                        choices=['naive', 'resnet', 'inspect'], default='naive')
    return parser.parse_args()


if __name__ == "__main__":

    print('Starting model...')
    args = parseArguments()

    orchestrate_preperation(args.model_name)
    orchestrate_model(args.model_name)
