import os
import shutil
import random

import numpy as np
import cv2 as cv


def data_folder_init(data_path, data_files_path, data_files_path_origin):
    '''
    Create train and remove duplicate files.
    Will only be executed once.
    '''
    if not os.path.exists(os.path.join(data_path, 'train')):
        shutil.move(os.path.join(data_files_path, 'train'),
                    os.path.join(data_path, 'train'))

    if os.path.exists(data_files_path):
        shutil.rmtree(data_files_path)


def data_folder_train_split(data_path, ratio_test_val, source_train_folder, target_test_val_folder):
    '''
    Split data into train, test and val (with pre-defined ratio).
    Will only be executed once.
    '''
    random.seed(4)

    if not os.path.exists(os.path.join(data_path, 'test')) and not os.path.exists(os.path.join(data_path, 'val')):
        for i in range(0, len(source_train_folder)):
            source = os.path.join(data_path, source_train_folder[i])
            target = os.path.join(data_path, target_test_val_folder[i])

            if not os.path.exists(target):
                os.makedirs(target)

            files = os.listdir(source)
            num_files_move = int(ratio_test_val[i] * len(files))
            files_move = random.sample(files, num_files_move)
            print(num_files_move)

            for file_name in files_move:
                source_f = os.path.join(source, file_name)
                target_f = os.path.join(target, file_name)
                shutil.move(source_f, target_f)


def read_training_data(data_directory, split, classes, img_size):
    '''
    Read training images and classes into multi-dimensional array.
    Images are compressed to img_size x img_size.
    '''
    split_data = [[], [], []]
    classes_data = [[], [], []]

    for spl_index, spl in enumerate(split):
        spl_path = os.path.join(data_directory, spl)
        for cla_index, cla in enumerate(classes):
            path = os.path.join(data_directory, spl, cla)
            if classes.index(cla) == 0:
                class_num = np.array([[1.], [0.]])
            else:
                class_num = np.array([[0.], [1.]])
            # class_num = classes.index(cla) # !!! replaced by if else statement
            for img in os.listdir(path):
                img_array = cv.imread(os.path.join(
                    path, img), cv.IMREAD_GRAYSCALE)
                img_array = cv.resize(img_array, (img_size, img_size))
                img_array = np.reshape(
                    img_array, (img_array.shape[0], img_array.shape[1], 1))
                img_array = img_array.astype("float32") / 255
                split_data[spl_index].append(img_array)
                classes_data[spl_index].append(class_num)
        print(spl_path, '(read', len(classes_data[spl_index]), 'images)')

    for i in range(0, len(split_data)):
        comb_list = list(zip(split_data[i], classes_data[i]))
        random.shuffle(comb_list)
        split_data[i], classes_data[i] = zip(*comb_list)

    return split_data, classes_data


def reshape_img(img):
    '''
    Reshape image to 3D array with third dimension as 1.
    '''
    return np.reshape(img, (img.shape[0], img.shape[1], 1))
