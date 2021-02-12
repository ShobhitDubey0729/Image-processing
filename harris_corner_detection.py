import cv2
import numpy as np

def harris_corner(input_img, window_size, threshold, k):
    output_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_GRAY2RGB)
    offset = int(window_size / 2)
    y_range = input_img.shape[0] - offset
    x_range = input_img.shape[1] - offset
    # first derivatives
    dy, dx = np.gradient(input_img)
    # second derivatives
    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2

    for y in range(offset, y_range):
        for x in range(offset, x_range):
            # Values of sliding window
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1

            # the variable names are representative to
            # the variable of the Harris corner equation
            windowIxx = Ixx[start_y: end_y, start_x: end_x]
            windowIxy = Ixy[start_y: end_y, start_x: end_x]
            windowIyy = Iyy[start_y: end_y, start_x: end_x]

            # Sum of squares of intensities of partial derevatives
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            # Calculate determinant and trace of the matrix
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy

            # Calculate corner response function(R) for Harris Corner equation
            r = det - k * (trace ** 2)
            if r > threshold:
                output_img[y, x] = (0, 0, 255)
    return [output_img]
