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

def confusion_matrix(cnn, x_test, y_test):
    '''
    Compute the confusion (error) matrix which has the following form:

       +-----------------+-----------------------+----------------------+
       |                 |  Predicted Matches    | Predicted NonMatches |
       +=================+=======================+======================+
       | True  Matches   | True Positives (TP)   | False Negatives (FN) |
       +-----------------+-----------------------+----------------------+
       | True NonMatches | False Positives (FP)  | True Negatives (TN)  |
       +-----------------+-----------------------+----------------------+
    '''
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    t_bar = tqdm(enumerate(zip(x_test, y_test)), total=len(x_test))

    for index, (img, img_label) in t_bar:
        result = pred(cnn, img)
        result_idx = np.argmax(result)
        img_label_idx = np.argmax(img_label)
        if result_idx == img_label_idx:
            if img_label_idx == 1:
                tp += 1
            else:
                tn += 1
        else:
            if img_label_idx == 1:
                fn += 1
            else:
                fp += 1
    
    return tp, tn, fn, fp


def accuracy(confusion_matrix)
    '''
    calculate the accuracy
    '''
    tp, tn, fn, fp = confusion_matrix(cnn, x_test, y_test)
    accuracy = float((tp + tn) / (tp + fp + fn + tn))

    return accuracy

def precision(confusion_matrix)
    '''
    calculate the precision
    '''
    tp, tn, fn, fp = confusion_matrix(cnn, x_test, y_test)
    precision = float(tp / (tp + fp))

    return precision

def recall(confusion_matrix)
    '''
    calculate the recall
    '''
    tp, tn, fn, fp = confusion_matrix(cnn, x_test, y_test)
    recall = float(tp / (tp + fn))

    return recall
