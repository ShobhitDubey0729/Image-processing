

from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage.io as io
import time


def compute_hist(image_path: Path, num_bins: int) -> list:
    # bins_vec and freq_vec should contain values computed by custom function
    # bins_vec_lib and freq_vec_lib should contain values computed by python library function
    k = 256 #no. of intensity values
    H =  [0]*num_bins
    B = []
    image1 = io.imread(image_path.as_posix())
    height, width = image1.shape[:2]
    bin_start = np.min(image1)
    bin_end = np.max(image1)
    step_size = (bin_end-bin_start)/bins
    for i in range(num_bins+1):
        if i == 0:
            B.append(bin_start)
        else:
            temp = B[i-1]
            temp = temp + step_size
            B.append(temp)
    for i in range(height):
        for j in range(width):
            x = image1[i, j] #getting the intensity value at location ith height and jth width from array
            a = x * (num_bins)/k
            a = int(a)
            H[a] = H[a] + 1
    bin_vec = B
    freq_vec = H
    freq_vec_lib, bin_vec_lib = np.histogram(image1, bins=num_)
    return [bins_vec, freq_vec, bins_vec_lib, freq_vec_lib]


def otsu_threshold(gray_image_path: Path) -> list:
    img = cv2.imread(gray_image_path.as_posix())
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.max() #it creates the normalised histogram
    Q = hist_norm.cumsum() 
    bins = np.arange(256)
    minimum =  np.inf
    maximum = -(np.inf)
    thr_w = None
    thr_b = None
    start_time_w = time.time()
    for i in range(0,256):
        prob1,prob2 = np.hsplit(hist_norm,[i+1]) # probabilities
        w0, w1 = Q[i],Q[255]-Q[i] #  classes
        b0,b1 = np.hsplit(bins,[i+1]) # weights
 # finding means and variances
        mean0,mean1 = np.sum(prob1*b0)/w0, np.sum(prob2*b1)/w1
        v0,v1 = np.sum(((b0-mean0)**2)*prob1)/w0,np.sum(((b1-mean1)**2)*prob2)/w1
 # calculates the minimization function
        var_w = w0*v0 + w1*v1
        if var_w < minimum:
            minimum = var_w
            thr_w = i
    end_time_w = time.time()
# calculation using between variance method
    start_time_b = time.time()
    for i in range(0,256):
        prob1,prob2 = np.hsplit(hist_norm,[i+1]) # probabilities
        w0, w1 = Q[i],Q[255]-Q[i] #  classes
        b0,b1 = np.hsplit(bins,[i+1]) # weights
 # finding means and variances
        mean0,mean1 = np.sum(prob1*b0)/w0, np.sum(prob2*b1)/w1   
        mean_total = w0*mean0+w1*mean1
        mean = mean_total
        var_b = w0*w1*((mean0-mean1))**2 #calculates the maximise function 
        if var_b > maximum:
            maximum = var_b
            thr_b = i
    end_time_b = time.time()
    time_w = end_time_w - start_time_w
    time_b = end_time_b - start_time_b
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            I = img[i, j] #intensity value 
            if I < thr_w:
                I = 0
                img[i, j]=I
                
            if I > thr_w:
                I = 255
                img[i, j]=I
    bin_image = io.imshow(img);
    return [thr_w, thr_b, time_w, time_b, bin_image]


def change_background(quote_image_path: Path, bg_image_path: Path) -> np.ndarray:
    quote = io.imread(quote_image_path.as_posix())
    background = io.imread(bg_image_path.as_posix())
    ret, otsu = cv2.threshold(quote,0,256,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = ret
    height, width = quote.shape[:2]
    for i in range(height):
        for j in range(width):
            I = quote[i, j] #intensity value 
            if I < thresh:
                I = 0
                quote[i, j]=I
                
            if I >thresh:
                I = 255
                quote[i, j]= 255
    img = cv2.addWeighted(quote, 0.4, background, 0.6, 0) #dst = src1*alpha + src2*beta + gamma
    modified_image = cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.distroyAllWindows()
    return modified_image


def count_connected_components(gray_image_path: Path) -> int:
    image2 = io.imread(gray_image_path.as_posix())
    k= num_characters =0
    height, width = image2.shape[:2]
    ret, otsu = cv2.threshold(image2,0,256,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = ret
    
    for i in range(height):
        for j in range(width):
            I = image2[i, j] #intensity value 
            if I < thresh:
                I = 0
                image2[i, j]=I
                
            if I >thresh:
                I = 255
                image2[i, j]=I
 # now our image has been binarized
    image2 = image2.astype('double') / 255
    R = np.zeros(shape=(height+1, width+1)) #region index number 
    def dfs(graph, i, j, k):
        if i<0 or i >= height or j<0 or j >= width or R[i, j] != 0 or graph[i, j] ==1:
            return;
        R[i, j] = k
        dfs(graph, i-1, j, k) #top
        dfs(graph, i, j-1, k) #left
        dfs(graph, i, j+1, k) #right
        dfs(graph, i+1, j, k) #down

    graph = image2
    for i in range(height):
        for j in range(width):
            if R[i, j] ==0 and graph[i, j] ==0:
                k = k+1
                dfs(graph, i, j, k)
#using library functions to know the small components like comma and dot.
    pixel_per_character = np.array(np.unique(R, return_counts=True)).T
    pixel_per_character = pixel_per_character[:,1].astype('int')
    plt.plot(pixel_per_character[1:])
# as we can see that only three component are there which are less than 100 vlue in the plot
    num_characters = k-3
    return num_characters


def binary_morphology(gray_image_path: Path) -> np.ndarray:
    image4 = io.imread(gray_image_path.as_posix())
    height, width = image4.shape[:2]
    ret, otsu = cv2.threshold(image4,0,256,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = ret 
    for i in range(height):
        for j in range(width):
            I = image4[i, j] #intensity value 
            if I < thresh:
                I = 0
                image4[i, j]=I
                
            if I >thresh:
                I = 255
                image4[i, j]=I
    new_image4 = np.zeros([height, width])
    for i in range(height-1):
        for j in range(width-1):
            window = [image4[i-1, j-1], 
                       image4[i-1, j], 
                       image4[i-1, j + 1], 
                       image4[i, j-1], 
                       image4[i, j], 
                       image4[i, j + 1], 
                       image4[i + 1, j-1], 
                       image4[i + 1, j], 
                       image4[i + 1, j + 1]] 
          
            window = sorted(window) 
            new_image4[i, j]= window[4] 

    new_image4 = new_image4.astype(np.uint8)
    cleaned_image = io.imshow(new_image4)
    return cleaned_image
