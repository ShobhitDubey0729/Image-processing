from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage.io as io


def binary_morphology(gray_image_path: Path) -> np.ndarray:
    image = io.imread(gray_image_path.as_posix())
    height, width = image.shape[:2]
    ret, otsu = cv2.threshold(image, 0, 256, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = ret

    for i in range(height):
        for j in range(width):
            I = image[i, j]  # intensity value
            if I < thresh:
                I = 0
                image[i, j] = I

            if I > thresh:
                I = 255
                image[i, j] = I

    new_image = np.zeros([height, width])

    for i in range(height - 1):
        for j in range(width - 1):
            window = [image[i - 1, j - 1],
                      image[i - 1, j],
                      image[i - 1, j + 1],
                      image[i, j - 1],
                      image[i, j],
                      image[i, j + 1],
                      image[i + 1, j - 1],
                      image[i + 1, j],
                      image[i + 1, j + 1]]

            window = sorted(window)
            new_image[i, j] = window[4]

    new_image = new_image.astype(np.uint8)
    cleaned_image = io.imshow(new_image)
    return cleaned_image
