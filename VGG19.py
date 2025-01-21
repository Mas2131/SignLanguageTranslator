""" 
This class implements the VGG19 architecture using a pretrained network.
Debugging prints have been commented out

Author: Anna Pia Mascolo 
"""
import os
import cv2
from tensorflow import keras
import keras
import numpy as np

# To use a pre-trained Network, saving time 
from keras.applications.vgg19 import VGG19
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.callbacks import Callback, ReduceLROnPlateau

# To evaluate the model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# To split the dataset
from sklearn.model_selection import train_test_split

# To show plots
import matplotlib.pyplot as plt                   

# For data augmentation
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


# An accuracy threshold help preventing overfitting
threshold = 98e-2

# Some constants
BATCH_SIZE = 32 #16 32 #64
EPOCHS_SIZE = 15 #10 15 #20
alpha = 1e-5 #Learning rate
IMG_SIZE = 224



#Preproses image: resize, reshape and normalize the images
def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

def reshape(img, n):
    img = img.reshape((n, IMG_SIZE,IMG_SIZE, 3)) 
    return img


def normalize(image):
    return image/255



#Load and split the dataset
def load_dataset(path):
    # Debugging 
    #for dirname, _, filenames in os.walk(PATH):
    #    for filename in filenames:
    #        print(os.path.join(dirname, filename))

    # Create lookup tables to map labels to integer (and viceversa)
    lookup = dict()
    reverselookup = dict()
    step = 0

    for i in os.listdir(path):
        #Ignore the readme.txt 
        if i != 'readme.txt':
            lookup[i] = step
            reverselookup[step] = i
            step += 1

    #reverselookup

    data = []
    labels = []

    n_data = 0

    for i in os.listdir(path):
        if i != 'readme.txt':
            step = 0
            for j in os.listdir(os.path.join(path, i)): 
                new_path = os.path.join(path, i, j) 
                img = preprocess(new_path)
                data.append(np.array(img))
                # print("Processed image shape: ", np.array(img).shape)
                step += 1
                
            label_val = np.full((step, 1), lookup[i]) 
            labels.append(label_val)
            n_data += step

    data = np.array(data, dtype = 'float32')
    labels = [label.flatten() for label in labels] 

    labels_new = np.concatenate(labels, axis = 0)
    labels_categorical = to_categorical(labels_new)
    reshaped_data = reshape(data, n_data)
    reshaped_data = normalize(reshaped_data)

    # Debugging
    # print("Data shape: ", data.shape)
    # print("Labels shape: ", labels_new.shape)
    # print("Sample lables (categorical): ", labels_categorical[:5])
    # print("Total images: ", len(data))
    # print("Sample image shape: ", data[0].shape)
    # print("Labels: ", np.unique(labels_new))
    
    # Split the data into two sets for training and testing. The testing set will later be split again for validation and testing
    # Training set: used for fitting the model. 60% of the dataset
    # Validation set: used to provide unbiased evaluation of the fitted model. 30% of the dataset
    # Testing set: used to test the unbiased estimation of the fitted model. 10% of the dataset

    train_data, test_data, train_label, test_label = train_test_split(reshaped_data, labels_categorical, test_size = 0.1, random_state=1, shuffle= True, stratify=labels_categorical)
    # print("Training data shape: ", train_data.shape, "Training labels shape: ", train_label.shape)
    # print("Testing data shape: ", test_data.shape, "Testing labels sape: ", test_label.shape)
    return train_data, test_data, train_label, test_label


# Generates, starting from the training dataset, a validation dataset. Images are modified (rotated, flipped, ...) to have a more robust training

def generators(train_images, train_labels, batch_size= BATCH_SIZE):
    # Define the augmentation parameters
    # Apply the augmentation to the training data
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        shear_range = 0.2,
        fill_mode = 'wrap',
        validation_split = 0.33
    )

    # Apply the augmentation to the training data
    train_generator = train_datagen.flow(
        train_images,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        subset='training'
    )

    validation_generator = train_datagen.flow(
        train_images,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        subset='validation'
    )

    return train_generator, validation_generator


# To prevent overfitting, the training is stopped when it reaches 98% of accuracy
class CallbackOverfitPrevention(Callback):
    def on_epoch_end(self, epoch, logs = None):
        if(logs.get('accuracy') >= threshold):
            print("\033[1;31mModel is probably overfitted. Training has stopped at epoch\033[0m", epoch)
            self.model.stop_training = True

# Save the model periodically (every 5 epochs)
class SaveEvery5Epochs(Callback):
    save_path = './VGG19_model_epoch_{epoch:02d}.weights.h5'
    save_freq = 5
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:  
            save_path = self.save_path.format(epoch=epoch + 1)
            self.model.save(save_path)
            

# With pretrained Network
class VGGNet19(keras.Model):
    def __init__(self, n_letters):
        super(VGGNet19, self).__init__()
        self.shape = (IMG_SIZE, IMG_SIZE, 3)
        self.letters = n_letters
        self.model = None
        self.build_model()
	
    def build_model(self):        
        # weights='imagenet' fetches the pretrained hyperparameters
        vgg19 = VGG19(weights= 'imagenet', input_shape=self.shape, classes= self.letters, include_top= False)
        
        # Freezes the last 10 layers so that are not retrained, saving time during training but still obaining good results
        for layer in vgg19.layers[:10]:
            layer.trainable = False
            
        
        # When working with a pretrained model, we only need to change the last layers
        x = Flatten()(vgg19.output)
        x = Dense(512, activation='relu', kernel_regularizer = l2(0.01))(x)
        
        # Added Batch normalization and an extra Dense layer compared to VGG16 to have better performances
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', kernel_regularizer = l2(0.01))(x)
        x = Dropout(0.5)(x)
        
        predictions = Dense(self.letters, activation='softmax')(x)

        self.model = Model(inputs=vgg19.input, outputs=predictions)
        
        # Used SGD as an optimizer
        self.model.compile(keras.optimizers.SGD(learning_rate= alpha, momentum = 0.9), loss = 'categorical_crossentropy', 
                           metrics =['accuracy',
                                     keras.metrics.FalsePositives(),
                                     keras.metrics.FalseNegatives(),
                                     keras.metrics.TruePositives(),
                                     keras.metrics.TrueNegatives(),
                                     keras.metrics.Recall(),
                                     keras.metrics.Precision()] )

        
    def training(self, train_data, train_labels, file_path):
        model = self.model 
        # Debugging
        # model.summary()
        
        # Using a variable learning rate to improve performances
        reduced_alpha = ReduceLROnPlateau(
            monitor='val_loss', 
            factor= 0.2,          
            patience = 3,          
            min_lr = 1e-6          
        )

        train_set, validation_set = generators(train_data, train_labels)
        
        fitted_model = model.fit(
            train_set,
            batch_size= BATCH_SIZE,
            epochs = EPOCHS_SIZE, 
            validation_data = validation_set,
            callbacks= [CallbackOverfitPrevention(), SaveEvery5Epochs(), reduced_alpha]
        )
        
        model.save(file_path)

        history = fitted_model.history

        # Plot training & validation accuracy
        plt.figure(figsize=(12, 6))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('VGG19 Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('VGG 19 Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Show plots
        plt.tight_layout()
        plt.savefig('./training_plot_VGG19.png')
        plt.show()
        return fitted_model
    
    def plot_confusion_matrix(self, y_true, y_pred, classes):
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, normalize="true")

        fig, ax = plt.subplots(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.BuPu, ax=ax, values_format=".2f")

        # Adjust font size for better readibility
        plt.setp(ax.get_xticklabels(), fontsize=10)  
        plt.setp(ax.get_yticklabels(), fontsize=10)  
        for labels in ax.texts:  
            labels.set_fontsize(8)
        plt.title("Normalized Confusion Matrix", fontsize = 16)

        # To prevent overlapping
        plt.tight_layout()
        
        plt.savefig('./CM_VGG19.png')
        plt.show()
        plt.close()
        
    
    def test(self, test_data, test_labels, classes):
        # y_prediction = self.model.predict(test_data)
        # Debugging
        # print("y_true:", test_labels)
        # print("classes:", classes)
        # print("y_pred:", y_prediction)
        y_pred = np.argmax(self.model.predict(test_data), axis=1)
        y_true = np.argmax(test_labels, axis=1)
        self.plot_confusion_matrix(y_true, y_pred, classes)
        print(classification_report(y_true, y_pred, target_names=classes))


    # Save the weights of the model
    def save(self, file_path):
        full_path = file_path + '_' + str(BATCH_SIZE) + '-' + str(EPOCHS_SIZE) + '.weights.h5'
        self.model.save_weights(full_path, overwrite = True)

    # Load the weights of the model for evaluation
    def load(file_path, n_letters):
        model = VGGNet19(n_letters)
        full_path = file_path + '_' + str(BATCH_SIZE) + '-' + str(EPOCHS_SIZE) + '.weights.h5'
        model.model.load_weights(full_path, skip_mismatch=False)
        model.model.compile(keras.optimizers.SGD(learning_rate= alpha, momentum = 0.9), loss = 'categorical_crossentropy', 
                           metrics =['accuracy',
                                     keras.metrics.FalsePositives(),
                                     keras.metrics.FalseNegatives(),
                                     keras.metrics.TruePositives(),
                                     keras.metrics.TrueNegatives(),
                                     keras.metrics.Recall(),
                                     keras.metrics.Precision()] )
        
        return model
