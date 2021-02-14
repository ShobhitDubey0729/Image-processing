# Implementation of adaptive Histogram equalization using 8-by-8 blocks of the given images

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt


def adaptive_hist_eq(image_path):
    img = io.imread(image_path)
    s1, s2 = img.shape
    A = s1 * s2
    # block_row = 8
    # block_col = 8
    a = 0
    for i in range(s1):
        for j in range(s2):
            x = img[i, j]
            b = np.cumsum(a)
            if x < 60:
                pass
            if x > 60:
                a = img[i, j] - 60
                img[i, j] = 60
    cl_img = b / (A * 255) + img

    # now we obtained clipped image to reduce noise after AHE
    for i in range(0, s1 - 8, 8):
        for j in range(0, s2 - 8, 8):
            patch = cl_img[i:i + 8, j:j + 8]
            # create 8 by 8 blocks
            H = [0.0] * 64
            for u in range(8):
                for v in range(8):
                    I = int(patch[u, v])
                    H[I] = H[I] + 1
            # to calculate cdf
            h = np.array(H)
            p = h / A
            cdf = np.cumsum(p)
            cdf = np.array(cdf)
            trans_fn = (256 * cdf)
            h, w = patch.shape
            temp_img = np.zeros((h, w))
            for w in range(h):
                for z in range(w):
                    temp_img[w, z] = trans_fn[int(patch[w, z])]
            cl_img[w:w + 8, z:z + 8] = temp_img
            # do the same for next block and so on
    cl_img = cl_img.astype('uint8')

    return [io.imshow(cl_img, cmap='gray')]
