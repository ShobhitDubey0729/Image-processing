# Implementation of Sobel Edge detector for pointing out edges in a given image

import numpy as np
import skimage.io as io
import cv2
import matplotlib.pyplot as plt


def sobel(image_path, threshold):
    image = io.imread(image_path)
    h, w = image.shape[:2]
    image = cv2.GaussianBlur(image, (5, 5), 0)
    Sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # function to convolve image with kernel
    def convolve(image, kernel):
        (iH, iW) = image.shape[:2]
        (kH, kW) = kernel.shape[:2]
        pad = (kW - 1) // 2
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        output = np.zeros((iH, iW), dtype="float32")
        for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
                k = (roi * kernel).sum()
                output[y - pad, x - pad] = k
        return output

    Dx = convolve(image, Sobel_x)
    Dy = convolve(image, Sobel_y)

    mag = np.zeros(image.shape)

    for i in range(h):
        for j in range(w):
            mag[i, j] = np.sqrt((Dx[i, j])**2 + (Dy[i, j])**2)
    mag[mag < threshold] = 0
    return [plt.imshow(mag, cmap='gray')]
