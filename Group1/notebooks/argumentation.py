import cv2 as cv
import numpy as np


img = cv.imread("Photos/park.jpg")

# Translation

"""
-x --> left
-y --> up
x --> right
y --> down

"""

def translate_image(image, x_shift, y_shift):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    translated_image = cv.warpAffine(image, M, (cols, rows))
    return translated_image

# Zoom
def zoom_image(image, zoom_factor):
    rows, cols = image.shape[:2]
    M = cv.getRotationMatrix2D((cols/2, rows/2), 0, zoom_factor)
    zoomed_image = cv.warpAffine(image, M, (cols, rows))
    return zoomed_image

zoomed = zoom_image(img, 3)
cv.imshow("zoomed", zoomed)


# Brightness
def adjust_contrast_brightness(image, alpha, beta):
    adjusted_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

# Noise
def add_gaussian_noise(image, mean, stddev):
    noisy_image = image.copy()
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
    noisy_image = cv.add(image, noise)
    return noisy_image

boise = add_gaussian_noise(img,3,3)
cv.imshow("noise",boise)

# Rotation
def rotate(image, angle, rotPoint=None):
    rows, cols = image.shape[:2]

    if rotPoint is None:
        rotPoint = (cols/2, rows/2)
    M = cv.getRotationMatrix2D(rotPoint,angle, 1.0)
    return cv.warpAffine(img, M, (rows, cols))

# Flip
def flip_image(image):
    flipped_image = cv.flip(image, 1)
    return flipped_image

cv.waitKey(0)


