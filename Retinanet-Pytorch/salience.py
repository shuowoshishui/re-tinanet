import sys
import os
import cv2
import numpy as np
from PIL import Image

import matplotlib.image as im
import matplotlib.pyplot as plt

from scipy.fftpack import dct, idct
import scipy.signal as nds
import scipy.misc as smi


def zz_gaussian2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def zz_dct2(blk):
    return dct(dct(blk.T, norm='ortho').T, norm='ortho')


def zz_idct2(blk):
    return idct(idct(blk.T, norm='ortho').T, norm='ortho')


def zz_salMap(img, ratio):
    # img = np.double(img)
    numChannels = img.shape[2]
    cSalMap = np.zeros([img.shape[0], img.shape[1], numChannels])
    outMap = np.zeros([img.shape[0], img.shape[1]])
    newMap = np.zeros([img.shape[0], img.shape[1]])
    finalMap = np.zeros([img.shape[0], img.shape[1]])
    finalSal = np.zeros([img.shape[0], img.shape[1], numChannels])

    for i in range(numChannels):
        cSalMap[:, :, i] = np.power(zz_idct2(np.sign(zz_dct2(img[:, :, i]))), 2)

    outMap = cSalMap.mean(axis=(2))

    kSize = img.shape[1] * 0.0450
    gKerVal = np.round(kSize * 4)
    gKer = zz_gaussian2D((gKerVal, gKerVal), kSize)
    newMap = nds.convolve(outMap, gKer)
    # print(img.shape[0],img.shape[1])
    # finalMap = smi.imresize(newMap, [720, 1080])

    finalMap = smi.imresize(newMap, [img.shape[0], img.shape[1]])
    print(np.max(outMap))
    print(np.max(gKer))
    print(np.max(newMap))
    print(np.max(finalMap))

    # print(np.max(finalMap),ratio)
    finalSal = np.multiply(img, finalMap[:, :, None]) * ratio  # 图片*saliencyMap*ratio得到结果

    clipped = np.clip(finalSal, 0., 1.)
    return clipped, finalMap


if __name__ == "__main__":

    data_path = '/home/chenshuo/data/images'
    img_list = os.listdir(data_path)

    if len(img_list) == 0:
        print('Data directory is empty.')
        exit()

    newImgDirectory = ('/home/chenshuo/data/images_salmap')
    if not os.path.exists(newImgDirectory):
        os.mkdir(newImgDirectory)

    count = 0
    for img_name in img_list:
        if img_name == '.gitkeep':
            continue
        if img_name.split('.')[-1] != 'png':
            continue
        img = im.imread(os.path.join(data_path, img_name))
        # cv2.imshow(img_name,img)
        # cv2.waitKey(10000)

        saliency, finalmap = zz_salMap(img, 1 / 100)

        # im.imsave(newImgDirectory+'/raw'+img_name,img)
        im.imsave(newImgDirectory + '/salmap' + img_name, finalmap, cmap="Greys_r", vmin=0, vmax=255)
        # im.imsave(newImgDirectory+'/'+img_name,finalmap)
        count += 1

