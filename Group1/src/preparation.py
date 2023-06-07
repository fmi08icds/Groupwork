import os
import shutil
import random


def data_folder_init(data_path, data_files_path, data_files_path_origin):
    '''
    create train and remove duplicate files
    '''
    if not os.path.exists(os.path.join(data_path, 'train')):
        shutil.move(os.path.join(data_files_path, 'train'),
                    os.path.join(data_path, 'train'))

    if os.path.exists(data_files_path):
        shutil.rmtree(data_files_path)


def data_folder_train_split(data_path, ratio_test_val, source_train_folder, target_test_val_folder):
    '''
    split data into train, test and val (with pre-defined ratio)
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
