# Implementation of Homomorphic filter for image enhancement
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import cv2


def generate_filter(img, gl, gh, D0, c):
    H = np.zeros(img.shape)
    P, Q = H.shape
    for i in range(P):
        for j in range(Q):
            D = np.sqrt(((i - P//2) ** 2 + (j - Q//2) ** 2))
            H[i, j] = (gh - gl) * (1 - (np.exp(-c*0.5*(D/D0)**2))) + gl
    return H
def homomorphic(img_path):
    img = io.imread(image_path)
    img = img/255
    im_log = np.log(img)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    mask = generate_filter(img, .2, 3, 30, 2)
    dft_shift_filt = np.multiply(mask, dft_shift)
    # shift origin from center to upper left corner
    back_ishift = np.fft.ifftshift(dft_shift_filt)
    # do idft saving as complex
    img_back = np.fft.ifft2(back_ishift)
    mag = np.real(img_back)
    img_homomorphic = np.exp(mag)
    norm_img = cv2.normalize(img_homomorphic, None, 1, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return [plt.imshow(norm_img, cmap='gray')]