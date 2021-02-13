"""
    This program implements different methods of denoising an images and comparing the
    results obtained. There are two parts 'a' and 'b'. In part a we are using Gaussian
    blur and Median blur, in part b used bilateral filter for the same which is very effective
    at edges.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def denoising(image_path, part):  # part a or b
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    if part == 1:
        # first filtering with Gaussian filter
        G_blur = cv2.GaussianBlur(img, (5, 5), 0)
        # now filtering with median filter(that uses laplacian)
        median_blur = cv2.medianBlur(img,9)
        plt.subplot(121), plt.imshow(median_blur, cmap='gray'), plt.title('Median')
        plt.subplot(122), plt.imshow(G_blur, cmap='gray'), plt.title('Gaussian')
        out = plt.show()
        return [out]
        """
            we see that using gaussian filter the edges are also getting blurred and 
            the image is not fully free from the impulsive noise
            but in median for kernel size greater than 3 noise has been removed completely
        """
    # 2_b: we will denoise the image using bilateral filtering technique
    if part == 2:
        def gaussian(x, sigma):
            return (1.0/(2*np.pi*(sigma**2)))*np.exp(-(x**2)/(2*(sigma**2)))

        def distance(x1, y1, x2, y2):
            return np.sqrt(np.abs((x1-x2)**2-(y1-y2)**2))

        def bilateral_filter(image, diameter, sigma_i, sigma_s):
            new_image = np.zeros(image.shape)

            for row in range(len(image)):
                for col in range(len(image[0])):
                    wp_total = 0
                    filtered_image = 0
                    for k in range(diameter):
                        for l in range(diameter):
                            n_x =row - (diameter/2 - k)
                            n_y =col - (diameter/2 - l)
                            if n_x >= len(image):
                                n_x -= len(image)
                            if n_y >= len(image[0]):
                                n_y -= len(image[0])
                            gi = gaussian(image[int(n_x)][int(n_y)] - image[row][col], sigma_i)
                            gs = gaussian(distance(n_x, n_y, row, col), sigma_s)
                            wp = gi * gs
                            filtered_image = filtered_image + (image[int(n_x)][int(n_y)] * wp)
                            wp_total = wp_total + wp
                    filtered_image = filtered_image // wp_total
                    new_image[row][col] = int(np.round(filtered_image))
            return new_image
        return bilateral_filter(img, 5, 0.2, 0.4)