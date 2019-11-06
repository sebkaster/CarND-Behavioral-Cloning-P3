import cv2
import numpy as np

def img_crop(img):
    img = img[40:135,:]
    return cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA)


def img_normalization(img):
    img = img / 127.5 - 1.0
    img = img.astype(np.float32)
    return img


def preprocess_img(img):
    img = img_normalization(img)
    return img_crop(cv2.cvtColor(img, cv2.COLOR_RGB2YUV))
