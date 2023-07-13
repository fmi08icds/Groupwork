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


class main_variables():
    '''
    Organize the variables for all classes
    '''

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
                               self.ratio_val, self.ratio_test, self.ratio_val]
        self.source_train_folder = ['train/NORMAL', 'train/NORMAL',
                                    'train/PNEUMONIA', 'train/PNEUMONIA']
        self.target_test_val_folder = ['test/NORMAL',
                                       'val/NORMAL', 'test/PNEUMONIA', 'val/PNEUMONIA']
        self.data_directory = 'data/'
        self.split = ['train', 'val', 'test']
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.img_size = 100
        self.epochs = 10
        self.learning_rate = 0.1
        self.batch_size = 64


def orchestrate_preperation(model_name):
    '''
    Orchestrate the preparation methods of the data
    '''
    args = main_variables(model_name)

    preparation.data_folder_init(
        args.data_path, args.data_files_path, args.data_files_path_origin)
    preparation.data_folder_train_split(
        args.data_path, args.ratio_test_val, args.source_train_folder, args.target_test_val_folder)


def exec_base_model(split_data, classes_data):
    '''
    Execute the base cnn model from model.py
    '''

    print('Executing base model...')
    args = main_variables('base')
    model.run_base_cnn(split_data, classes_data,
                       args.epochs, args.learning_rate)


def exec_torch_model(split_data, classes_data):
    '''
    Execute the torch model from model.py
    '''

    print('Executing pytorch model...')
    args = main_variables('torch')
    model.run_torch_cnn(split_data, classes_data,
                        args.epochs, args.learning_rate, args.batch_size)


def exec_inspection(split_data, classes_data):
    '''
    image selection and show for testing purposes from util.py
    '''

    print('Executing image inspection...')
    args = main_variables('inspect')

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
        args.data_directory, args.split, args.classes, args.img_size, model_name)

    all_count_split_data = 0
    for i in split_data:
        all_count_split_data += len(i)
    print('All images: %s' % (all_count_split_data))

    if model_name == 'base':  # run the base cnn model
        exec_base_model(split_data, classes_data)

    elif model_name == 'torch':  # run the torch model
        exec_torch_model(split_data, classes_data)

    elif model_name == 'inspect':  # run a sample image
        exec_inspection(split_data, classes_data)

    print('Finished %s model.' % (model_name))


def parseArguments():
    '''
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', help='model to use',
                        choices=['base', 'torch', 'inspect'], default='base')
    return parser.parse_args()


if __name__ == "__main__":

    print('Starting model...')
    args = parseArguments()

    orchestrate_preperation(args.model_name)
    orchestrate_model(args.model_name)
