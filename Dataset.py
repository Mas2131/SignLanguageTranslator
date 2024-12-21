"""
This file opens the dataset and divides it into training,
validating and testing set.

Author: Anna Pia Mascolo
"""

import os
import numpy as np                   
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical



import preprocess_utils 


PATH = './archive/LIS-fingerspelling-dataset/'

#My PC is not cuda compatible, therefore I disabled all the GPU related warning
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_dataset():
    # Debugging 
    #for dirname, _, filenames in os.walk(PATH):
    #    for filename in filenames:
    #        print(os.path.join(dirname, filename))

    lookup = dict()
    reverselookup = dict()
    step = 0

    for i in os.listdir(PATH):
        #Ignore the readme.txt 
        if i != 'readme.txt':
            lookup[i] = step
            reverselookup[step] = i
            step += 1

    #reverselookup

    data = []
    labels = []

    n_data = 0
    for i in os.listdir(PATH):
        if i != 'readme.txt':
            step = 0
            for j in os.listdir(PATH + str(i)):
                new_path = PATH+ str(i) + '/' + str(j)
                img = preprocess_utils.preprocess(new_path)
                # Debug
                # print("Processed image shape: ", np.array(img).shape)
                data.append(np.array(img))
                step += 1
            
            label_val = np.full((step, 1), lookup[i]) 
            labels.append(label_val)
            # Debug
            #print("Step: ", step, "Labels shape: ", labels[-1].shape)
            n_data += step

    data = np.array(data, dtype = 'float32')
    labels = [label.flatten() for label in labels] 

    labels_new = np.concatenate(labels, axis = 0)
    labels_categorical = to_categorical(labels_new)
    reshaped_data = preprocess_utils.reshape(data, n_data)
    reshaped_data = preprocess_utils.normalize(reshaped_data)

    # Debug
    #print("Data shape: ", data.shape)
    #print("Labels shape: ", labels_new.shape)
    #print("Sample lables (categorical): ", labels_categorical[:5])


    #Training set: used for fitting the model. 60% of the dataset
    #Validation set: used to provide unbiased evaluation of the fitted model. 30% of the dataset
    #Testing set: used to test the unbiased estimation of the fitted model. 10% of the dataset

    train_data, test_data, train_label, test_label = train_test_split(reshaped_data, labels_categorical, test_size = 0.2, random_state=1, shuffle= True, stratify=labels_categorical)

    return train_data, test_data, train_label, test_label
