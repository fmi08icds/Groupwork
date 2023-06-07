import os

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

if __name__ == "__main__":
    preparation.data_folder_init(
        data_path, data_files_path, data_files_path_origin)
    preparation.data_folder_train_split(
        data_path, ratio_test_val, source_train_folder, target_test_val_folder)
