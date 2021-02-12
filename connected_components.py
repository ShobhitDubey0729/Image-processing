from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage.io as io


def count_connected_components(gray_image_path: Path) -> int:
    image2 = io.imread(gray_image_path.as_posix())
    k = 0  # number of characters
    height, width = image2.shape[:2]
    ret, otsu = cv2.threshold(image2, 0, 256, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = ret

    for i in range(height):
        for j in range(width):
            I = image2[i, j]  # intensity value
            if I < thresh:
                I = 0
                image2[i, j] = I

            if I > thresh:
                I = 255
                image2[i, j] = I
    # now our image has been binarized
    image2 = image2.astype('double') / 255
    R = np.zeros(shape=(height + 1, width + 1))  # region index number

    def dfs(graph, i, j, k):
        if i < 0 or i >= height or j < 0 or j >= width or R[i, j] != 0 or graph[i, j] == 1:
            return
        R[i, j] = k
        dfs(graph, i - 1, j, k)  # top
        dfs(graph, i, j - 1, k)  # left
        dfs(graph, i, j + 1, k)  # right
        dfs(graph, i + 1, j, k)  # down

    graph = image2
    for i in range(height):
        for j in range(width):
            if R[i, j] == 0 and graph[i, j] == 0:
                k = k + 1
                dfs(graph, i, j, k)
    # using library functions to know the small components like comma and dot.
    pixel_per_character = np.array(np.unique(R, return_counts=True)).T
    pixel_per_character = pixel_per_character[:, 1].astype('int')
    plt.plot(pixel_per_character[1:])
    # as we can see that only three component are there which are less than 100 value in the plot
    num_characters = k - 3  # K-N if 'N' is the not connected as will be displayed in figure
    return num_characters
