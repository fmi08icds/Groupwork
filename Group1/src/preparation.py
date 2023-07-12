import os
import shutil
import random
import cv2 as cv
import numpy as np


def data_folder_init(data_path, data_files_path, data_files_path_origin):
    '''
    create train and remove duplicate files
    '''
    if not os.path.exists(os.path.join(data_path, 'train')):
        shutil.move(os.path.join(data_files_path, 'train'), os.path.join(data_path, 'train'))

    if os.path.exists(data_files_path_origin):
        shutil.rmtree(data_files_path_origin)


def data_folder_train_split(data_path, ratio_test_val, source_train_folder, target_test_val_folder):
    '''
    split data into train, test and val (with pre-defined ratio)
    '''
    random.seed(4)

    if not os.path.exists(os.path.join(data_path, 'test')) and not os.path.exists(os.path.join(data_path, 'val')):
        for i in range(len(source_train_folder)):
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


# Translation
def translate_image(image, x_shift, y_shift):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    translated_image = cv.warpAffine(image, M, (cols, rows))
    return translated_image


# Zoom
def zoom_image(image, zoom_factor):
    rows, cols = image.shape[:2]
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 0, zoom_factor)
    zoomed_image = cv.warpAffine(image, M, (cols, rows))
    return zoomed_image


# Rotation
def rotate(image, angle, rotPoint=None):
    rows, cols = image.shape[:2]

    if rotPoint is None:
        rotPoint = (cols / 2, rows / 2)
    M = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    return cv.warpAffine(image, M, (cols, rows))


def apply_random_transformations(folder_path):
    # Liste aller Dateien im angegebenen Ordner
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        # Überprüfen, ob es sich um eine Bilddatei handelt
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(folder_path, file_name)

            # Bild laden
            image = cv.imread(file_path)

            # Zufällige Anzahl an Transformationen auswählen
            num_transformations = random.randint(1, 3)

            for _ in range(num_transformations):
                # Zufällige Transformation auswählen
                transformation = random.choice(['translate', 'zoom', 'rotation'])

                if transformation == 'translate':
                    # Zufällige Verschiebungswerte auswählen
                    x_shift = random.randint(-50, 50)
                    y_shift = random.randint(-50, 50)
                    image = translate_image(image, x_shift, y_shift)

                elif transformation == 'zoom':
                    # Zufälliger Zoom-Faktor auswählen
                    zoom_factor = random.uniform(1.0, 2.0)
                    image = zoom_image(image, zoom_factor)

                elif transformation == 'rotation':
                    # Zufälligen Rotationswinkel auswählen
                    angle = random.randint(-45, 45)
                    image = rotate(image, angle)


folder_path = "path_to_folder"  # Provide the path to the folder
apply_random_transformations(folder_path)
