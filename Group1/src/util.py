import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def select_random_image(split_data, classes_data, split_set):
    '''
    choose a random image for testing purposes
    '''
    rand_img_num = np.random.randint(0, len(split_data[split_set]))
    img = split_data[split_set][rand_img_num]
    img_label = classes_data[split_set][rand_img_num]
    return img, img_label, rand_img_num


def show_random_image(classes, classes_data, img, rand_img_num, split_set=0):
    '''
    show a random image for testing purposes
    '''
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(classes[classes_data[split_set][rand_img_num]])
    plt.show()
