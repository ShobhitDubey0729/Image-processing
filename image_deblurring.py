# This program is for images deblurring using different techniques

import numpy as np
import skimage.io as io
import scipy.io as sio

# we need a blur kernel for deblurring implementation
Kernel_path = 'kernel_path'
h = sio.loadmat(Kernel_path)
kernel_mat = h['h']

def deblur(image_path, filter_name, sigma, blur_kernel):
    image = io.imread(image_path)
    shape = image.shape
    
    if filter_name == 'inverse':
        window = np.zeros((shape))
        window[:19, :49] = blur_kernel
        H = np.fft.fft2(window)
        fft_image = np.fft.fft2(image)
        norm_H = H / np.abs(H.max())
        norm_image = fft_image / np.abs(fft_image.max())
        inv_H = np.linalg.inv(norm_H)
        inv_convol = np.multiply(inv_H, norm_image)
        inv_convol = inv_convol / np.abs(inv_convol.max())
        inv_con_image = inv_convol * np.abs(fft_image.max())
        inverse_fft = np.fft.ifft2(inv_con_image)
        output = ((np.abs(inverse_fft)) + 0.5).astype('uint8')
        return [output]

    if filter_name == 'wiener':
        window = np.zeros((shape))
        window[:19, :49] = blur_kernel
        H = np.fft.fft2(blur_kernel, image.shape)
        fft_image = np.fft.fft2(image)
        sigma2 = np.power(sigma, 2)
        conj_H = np.conj(H)
        kernel = conj_H / (np.abs(H) ** 2 + sigma)
        fft_image = fft_image * kernel
        inverse_fft = np.fft.ifft2(fft_image)
        output = np.abs(inverse_fft)
        return [output]

    if filter_name == 'constrained_least-square':
        G = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        G = np.fft.fft2(G, image.shape)
        G = np.abs(G ** 2)
        H = np.fft.fft2(blur_kernel, image.shape)
        fft_image = np.fft.fft2(image)
        conj_H = np.conj(H)
        H2 = np.abs(H ** 2)
        num = fft_image * conj_H
        den = H2 + (0.1 * G)
        F_hat = num / den
        inverse_fft = np.fft.fft2(F_hat)
        output = ((np.abs(inverse_fft)) + 0.5).astype('uint8')
        return [output]
