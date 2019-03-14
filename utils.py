import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np
import cv2

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=96, hrg=96, is_random=is_random)
    # x = x / (255. / 2.)
    # x = x - 1.
    # x = (x - 0.5)*2
    return x

def rescale(x):
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    # print("channel: ", x.shape[-1])
    x = imresize(x, size=[24, 24], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    # x = (x - 0.5)*2
    return x

def odbtc(img):
    # print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = np.uint8(img)
    # print(img.shape)
    pattern = np.array([[1,17,5,21,2,18,6,22],
        [25,9,29,13,26,10,30,14],
        [7,23,3,19,8,24,4,20],
        [31,15,27,11,32,16,28,12],
        [2,18,6,22,1,17,5,21],
        [26,10,30,14,25,9,29,13],
        [8,24,4,20,7,23,3,19],
        [32,16,28,12,31,15,27,11]])

    pattern = pattern.astype(float)

    pattern = pattern / 32

    pattern = (pattern - pattern.min()) / (pattern.max()-pattern.min())
    # pattern = cv2.imread('img/pattern/bayer16.png',0)

    img = (img - img.min()) / (img.max()-img.min())
    height, width = img.shape[:2]
    pheight, pwidth = pattern.shape[:2]
    bs = pheight
    lheight = int(height / bs)
    lwidth = int(width / bs)

    img = img.astype(float)

    # imgnew = np.empty((height,width))

    for i in range(0,lheight):
        for j in range(0,lwidth):
            tempImg = img[i*bs:i*bs+bs,j*bs:j*bs+bs]
            nMax = np.amax(tempImg)
            nMin = np.amin(tempImg)
            k = nMax - nMin
            for x in range(0,bs):
                for y in range(0,bs):
                    try:
                        DA = k * pattern[x][y]
                        th = DA + nMin
                        if img[i*bs+x][j*bs+y] >= th:
                            img[i*bs+x][j*bs+y] = nMax
                        else:
                            img[i*bs+x][j*bs+y] = nMin
                    except IndexError:
                        pass

    img = ((img - img.min()) / (img.max()-img.min()) * 255).astype('int')
    # img = np.expand_dims(img, axis=0)
    # print(img.shape)
    img = np.uint8(img)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = img / (255. / 2.)
    img = img - 1.
    # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # print(img.shape)
    return img