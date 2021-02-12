from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage.io as io
import time


def compute_hist(image_path: Path, num_bins: int) -> list:
    k = 256  # no. of intensity values
    H = [0] * num_bins
    B = []
    image1 = io.imread(image_path.as_posix())
    height, width = image1.shape[:2]
    bin_start = np.min(image1)
    bin_end = np.max(image1)
    step_size = (bin_end - bin_start) / num_bins

    for i in range(num_bins + 1):
        if i == 0:
            B.append(bin_start)
        else:
            temp = B[i - 1]
            temp = temp + step_size
            B.append(temp)

    for i in range(height):
        for j in range(width):
            x = image1[i, j]
            # getting the intensity value at location ith height and jth width from array
            a = x * num_bins / k
            a = int(a)
            H[a] = H[a] + 1

    bin_vec = B
    freq_vec = H
    freq_vec_lib, bin_vec_lib = np.histogram(image1, bins=num_bins)
    return [bin_vec, freq_vec, bin_vec_lib, freq_vec_lib]