from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage.io as io
import time


def otsu_threshold(gray_image_path: Path) -> list:
    img = cv2.imread(gray_image_path.as_posix())
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.max()  # it creates the normalised histogram
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    minimum = np.inf
    maximum = -np.inf
    thr_w = None
    thr_b = None

    start_time_w = time.time()

    for i in range(0, 256):
        prob1, prob2 = np.hsplit(hist_norm, [i + 1])  # probabilities
        w0, w1 = Q[i], Q[255] - Q[i]  # classes
        b0, b1 = np.hsplit(bins, [i + 1])  # weights
        # finding means and variances
        mean0, mean1 = np.sum(prob1 * b0) / w0, np.sum(prob2 * b1) / w1
        v0, v1 = np.sum(((b0 - mean0) ** 2) * prob1) / w0, np.sum(((b1 - mean1) ** 2) * prob2) / w1
        # calculates the minimization function
        var_w = w0 * v0 + w1 * v1

        if var_w < minimum:
            minimum = var_w
            thr_w = i

    end_time_w = time.time()

    # calculation using between variance method
    start_time_b = time.time()

    for i in range(0, 256):
        prob1, prob2 = np.hsplit(hist_norm, [i + 1])  # probabilities
        w0, w1 = Q[i], Q[255] - Q[i]  # classes
        b0, b1 = np.hsplit(bins, [i + 1])  # weights
        # finding means and variances
        mean0, mean1 = np.sum(prob1 * b0) / w0, np.sum(prob2 * b1) / w1
        mean_total = w0 * mean0 + w1 * mean1
        mean = mean_total
        var_b = w0 * w1 * (mean0 - mean1) ** 2  # calculates the maximise function
        if var_b > maximum:
            maximum = var_b
            thr_b = i

    end_time_b = time.time()
    # calculating which method is faster
    time_w = end_time_w - start_time_w
    time_b = end_time_b - start_time_b
    height, width = img.shape[:2]

    for i in range(height):
        for j in range(width):
            I = img[i, j]  # intensity value
            if I < thr_w:
                I = 0
                img[i, j] = I

            if I > thr_w:
                I = 255
                img[i, j] = I
    bin_image = io.imshow(img)
    return [thr_w, thr_b, time_w, time_b, bin_image]
