""" 
This module has some utils for preprocessing the images. 
Author: Mascolo Anna Pia 
"""

import cv2
from PIL import Image



IMG_SIZE = 150

#Transform the image into a grayscale one to make the training faster
#Resize and normalize the images
def preprocess(image_path):
    img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    return img

def reshape(img, n):
    img = img.reshape((n, IMG_SIZE,IMG_SIZE, 3)) 
    return img

def normalize(image):
    return image/255