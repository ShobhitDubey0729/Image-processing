#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import math


# In[64]:


#question1a DIP_A2
image1_path = 'Data/images/LowLight_1.png'
image2_path = 'Data/images/LowLight_2.png'
image1 = io.imread(image1_path)
image2 = io.imread(image2_path)
max_1 = np.max(image1)
max_2 = np.max(image2)
gain1, gain2 =int(255/max_1), int(255/max_2)
new_image1 = image1*gain1
new_image2 = image2*gain2
plt.subplot(221);
io.imshow(image1);
plt.subplot(222);
io.imshow(new_image1);
plt.subplot(223);
io.imshow(image2);
plt.subplot(224);
io.imshow(new_image2);
max_1, max_2


# In[65]:


#question1b DIP_A2
image1_path = 'Data/images/LowLight_1.png'
image2_path = 'Data/images/LowLight_2.png'
image3_path = 'Data/images/Hazy.png'
image1 = io.imread(image1_path)
image2 = io.imread(image2_path)
image3 = io.imread(image3_path)
h1, w1 = image1.shape
h2, w2 = image2.shape
h3, w3 = image3.shape
norm_image1 = image1/255
norm_image2 = image2/255
norm_image3 = image3/255
for i in range(h1):
    for j in range(w1):
        x = norm_image1[i, j]
        norm_image1[i, j] = x**.4
for i in range(h2):
    for j in range(w2):
        x = norm_image2[i, j]
        norm_image2[i, j] = x**.5
for i in range(h3):
    for j in range(w3):
        x = norm_image3[i, j]
        norm_image3[i, j] = x**2.5 #since its hazy so if we put gamma less than 1 then it will be more hazy
plt.subplot(231);
io.imshow(image1);
plt.subplot(232);
io.imshow(image2);
plt.subplot(233);
io.imshow(image3);
plt.subplot(234);
io.imshow(norm_image1, cmap = 'gray');
plt.subplot(235);
io.imshow(norm_image2, cmap = 'gray');
plt.subplot(236);
io.imshow(norm_image3, cmap = 'gray');


# In[66]:


#question1c DIP_A2
def hist_eq(image_path): 
    image = io.imread(image_path)
    height, width = image.shape
    A = height * width
    H = [0.0]*256
    for i in range(height):
        for j in range(width):
            I = image[i, j]
            H[I] = H[I]+1
#now to calculate p(k)
    h =np.array(H)
    p = h/A
    for i in range(len(H)):
        cdf = np.cumsum(p)
    cdf = np.array(cdf)
    trans_fn = (255*cdf)
    trans_img = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            trans_img[i, j] = trans_fn[image[i, j]]
# now for plotting the images
    plt.subplot(221)
    a = io.imshow(image, cmap='gray')
    plt.subplot(222)
    b = io.imshow(trans_img, cmap='gray')
    plt.subplot(223)
    c = plt.hist(image)
    plt.subplot(224)
    d = plt.hist(trans_img)
    return a, b, c, d


# In[67]:


#question1d DIP_A2
image_path = 'Data/images/StoneFace.png'
img = io.imread(image_path)
s1, s2 = img.shape
A = s1*s2
block_row = 8
block_col = 8
a = 0
for i in range(s1):
    for j in range(s2):
        x = img[i, j]
        b = np.cumsum(a)
        if x<60:
            pass
        if x>60:
            a = img[i, j]-60
            img[i, j] = 60
cl_img = b/(A*255)+img    #now we obtained clipped image to reduce noise after AHE       
for i in range(0,s1-8,8):
    for j in range(0,s2-8,8):
        patch = cl_img[i:i+8, j:j+8]
        #create 8 by 8 blocks 
        H = [0.0]*64
        for i in range(8):
            for j in range(8):
                I = int(patch[i,j])
                H[I] = H[I] + 1
        #to calculate cdf
        h = np.array(H)
        p = h/A
        cdf = np.cumsum(p)
        cdf = np.array(cdf)
        trans_fn = (256*cdf)
        h, w = patch.shape
        temp_img = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                temp_img[i, j] = trans_fn[int(patch[i, j])]
        cl_img[i:i+8, j:j+8] = temp_img #do the same for next block and so on 
            
        
io.imshow(cl_img, cmap='gray');



# In[72]:


#question2 DIP_A2
image_path = 'Data/images/MathBooks.png'
image = io.imread(image_path)
#Split into RGB Channels
Red = image[:,:,0]
Green = image[:,:,1]
Blue = image[:,:,2]
r1, r2 = Red.shape
#since shape of all colour channels will be same
new_Red = np.copy(Red)
new_Green = np.copy(Green)
new_Blue = np.copy(Blue)

for i in range(r1):
    for j in range(r2):
        if Red[i, j]>=0 and Red[i, j]<=10 and Green[i, j]>=0 and Green[i, j]<=10 and Blue[i, j]>=0 and Blue[i, j]<=50:
            new_Red[i, j] = 0
            new_Green[i, j] = 0
            new_Blue[i, j] = 0
        if Red[i, j]>=75 and Red[i, j]<=255 and Green[i, j]>=60 and Green[i, j]<=255 and Blue[i, j]>=245 and Blue[i, j]<=255:
            new_Red[i, j] = 255
            new_Green[i, j] = 255
            new_Blue[i, j] = 255
        else:
            max_Red = 74
            gain_r = 255/max_Red
            new_Red[i, j] = Red[i, j]*gain_r
            max_Green = 59
            gain_g = 255/max_Green
            new_Green[i, j] = Green[i, j]*gain_g
            max_Blue = 244
            gain_b = 255/max_Blue
            new_Blue[i, j] = Blue[i, j]*gain_b
img = np.copy(image)

img[:,:,0] = new_Red
img[:,:,1] = new_Green
img[:,:,2] = new_Blue

plt.subplot(121);
io.imshow(image);
plt.subplot(122);
io.imshow(img);


# In[69]:


#question3
def resize(image, factor):
    b = factor
    img = io.imread(image)
    s1, s2 = img.shape
    J = np.zeros((int(s1*b), int(s2*b)))
    h, w = J.shape
    for i in range(h-1):
        for j in range(w-1):
            a1, a2 = int(0.5+i/b), int(0.5+j/b) #using the nearest neighbour
            I = img[a1, a2]
            J[i, j]=I
    res_img = io.imshow(J, cmap='gray')
    return res_img   


# In[71]:


#question4
def ImgRotate(image, x):#degree of ratation is x here
    img = io.imread(image)
    s1, s2 = img.shape
    h = int(np.absolute(s1*(np.cos(x*(np.pi/180)))+s2*(np.sin(x*(np.pi/180)))))
    w = int(np.absolute(s1*(np.sin(x*(np.pi/180)))+s2*(np.cos(x*(np.pi/180)))))
    J = np.zeros((h, w))
    for i in range(s1-1):
        for j in range(s2-1):
            a1 = int((i*(np.cos(x*(np.pi/180)))+j*(np.sin(x*(np.pi/180)))))
            a2 = int((-i*(np.sin(x*(np.pi/180)))+j*(np.cos(x*(np.pi/180)))))
            if a1<288 and a2<384:
                J[i, j] = img[a1, a2]
    rot_img = io.imshow(J, cmap='gray')
    
    return rot_img
    

    
    
    

