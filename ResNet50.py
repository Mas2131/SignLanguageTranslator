""" 
This code implements a custom ResNet50-based image classification framework.

The dataset is split into 60% training, 20% validation, and 20% testing. 
The resulting split dataset is stored in a new directory instead of overwriting the unsplit one.

The folowing plots are shown:
- Training and validation accuracy/loss across epochs
- Confusion matrix with class-wise predictions

Debugging prints have been commented out

Author: Ida Perfetto 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential, Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications import ResNet50 # Import the ResNet50 model from Keras applications
from tensorflow.keras.applications.resnet import preprocess_input # Import the preprocess_input function for ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns # For more complex and informative plots
import time
import shutil # for copy and moving files

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Remove this line to use GPU

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 2 # Choose between 16, 32 and 64
EPOCHS = 1 # Choose between 10, 15 and 20
SEED = 42  # Seed for reproducibility of results
THRESHOLD = 0.98 # Treshold to prevent overfitting
#DATASET_TRESHOLD = 500 # Average images per class treshold

# To prevent overfitting, the training is stopped when it reaches 98% of accuracy
class CallbackOverfitPrevention(Callback):
    def on_epoch_end(self, epoch, logs = None):
        if logs.get('accuracy', 0) >= THRESHOLD or logs.get('val_accuracy', 0) >= THRESHOLD:
            print("\033[1;31mModel is probably overfitted. Training has stopped\033[0m") # 033 for bold text, 31 for red text 
            self.model.stop_training = True 


class ResNet:
    def __init__(self, num_classes, dataset_path):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.dataset_path = dataset_path
        self.model = None # Placeholder for the model
        self.build_model() # Build the ResNet model
        self.split_dataset_path = self.get_split_dataset_path()  # Initialize split dataset path here
        self.train_dir, self.val_dir, self.test_dir = None, None, None # Initialize the variables that will contain the path for the splitted dataset
        self.split_sets = False # Flag to indicate whether the dataset is split
        self.LEARNING_RATE = 0

    # Construct the path for the split dataset folder
    def get_split_dataset_path(self):
        dataset_folder_name = os.path.basename(self.dataset_path) # Get the dataset folder name
        dataset_parent_dir = os.path.dirname(self.dataset_path) # Get the parent directory
        split_dataset_path = os.path.join(dataset_parent_dir, f"{dataset_folder_name}_split") # Define split dataset path
        return split_dataset_path

    
    def build_model(self):

        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), classes=self.num_classes)


        # Determine dataset type to adjust training parameters
        dataset_folder_name = os.path.basename(self.dataset_path)

        # Adjusts the model configuration (frozen and unfrozen layers, dense layer, learning rate) based on the dataset type
        if dataset_folder_name == 'BSL(1000)':
            for layer in base_model.layers:
                if "BatchNormalization" not in layer.__class__.__name__:
                    layer.trainable = False
            print('BatchNormalization layers unforzen') 
            Dense_layer = Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))
            print('Dense layer with regularizer')
            self.LEARNING_RATE = 1e-5
            print(f'Learning rate = {self.LEARNING_RATE}')
        
        elif dataset_folder_name == 'BSL(300)':
            for layer in base_model.layers:
                if "BatchNormalization" not in layer.__class__.__name__:
                    layer.trainable = False
            print('BatchNormalization layers unforzen') 
            Dense_layer = Dense(512, activation='relu')
            print('Dense layer without regularizer')
            self.LEARNING_RATE = 1e-4
            print(f'Learning rate = {self.LEARNING_RATE}')
        
        elif dataset_folder_name == 'ASL(1000)':
            for layer in base_model.layers:
                if "BatchNormalization" not in layer.__class__.__name__:
                    layer.trainable = False
            print('BatchNormalization layers unforzen') 
            Dense_layer = Dense(512, activation='relu')
            print('Dense layer without regularizer')
            self.LEARNING_RATE = 1e-5
            print(f'Learning rate = {self.LEARNING_RATE}')
        
        elif dataset_folder_name == 'ASL(300)' or dataset_folder_name == 'LIS(1000)' or dataset_folder_name == 'LIS(300)' or dataset_folder_name == 'Combined':
            for layer in base_model.layers:
                if "BatchNormalization" not in layer.__class__.__name__:
                    layer.trainable = False
            print('All base model layers forzen')
            Dense_layer = Dense(512, activation='relu')
            print('Dense layer without regularizer')
            self.LEARNING_RATE = 1e-4
            print(f'Learning rate = {self.LEARNING_RATE}')
        

        # Build and compile the final ResNet model with custom top layers
        x = GlobalAveragePooling2D()(base_model.output) # GlobalAveragePooling to reduce overfitting
        x = Dense_layer(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=outputs)
        
        
        self.model.compile(optimizer=Adam(learning_rate=self.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model
    

    # Creates path for training, validation, and test datasets
    def split_dataset(self):
        print("\033[1;36mDataset splitting started...\033[0m") # 033 for bold text, 36 for cyan text
        
        train_dir = os.path.join(self.split_dataset_path, 'train')
        val_dir = os.path.join(self.split_dataset_path, 'val')
        test_dir = os.path.join(self.split_dataset_path, 'test')
        
        if os.path.exists(self.split_dataset_path):
            shutil.rmtree(self.split_dataset_path)
            
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for class_name in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_name)
            if os.path.isdir(class_path):
                images = os.listdir(class_path)
                #print(f"Class: {class_name}, Number of images: {len(images)}") # DEBUG PRINT
                #if len(images) == 0: # DEBUG PRINT
                    #print(f"Warning: Class {class_name} has no images.") # DEBUG PRINT
                train_images, test_val_images = train_test_split(images, test_size=0.4, random_state=SEED)
                val_images, test_images = train_test_split(test_val_images, test_size=0.25, random_state=SEED)
                
                os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
                os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
                os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

                for image in train_images:
                    shutil.copy(os.path.join(class_path, image), os.path.join(train_dir, class_name, image))
                for image in val_images:
                    shutil.copy(os.path.join(class_path, image), os.path.join(val_dir, class_name, image))
                for image in test_images:
                     shutil.copy(os.path.join(class_path, image), os.path.join(test_dir, class_name, image))
        print(f"\033[1;36mDataset splitting completed. You can find the split dataset in: {self.split_dataset_path}\033[0m") # 033 for bold text, 36 for cyan text
        return train_dir, val_dir, test_dir
    

    def train(self, model_save_path):

        train_dir, val_dir, test_dir = None, None, None
        
        if not self.split_sets: # Check if dataset is already split
            if os.path.exists(self.split_dataset_path): # Check if split directory exists
                print("\033[1;34mLoading dataset...\033[0m")
                train_dir = os.path.join(self.split_dataset_path, 'train')
                val_dir = os.path.join(self.split_dataset_path, 'val')
                test_dir = os.path.join(self.split_dataset_path, 'test')
            else:
               train_dir, val_dir, test_dir = self.split_dataset() # Perform dataset splitting
            self.split_sets = True # Set the flag
        
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

        train_dir = self.train_dir
        val_dir = self.val_dir
        test_dir = self.test_dir

        #print(f"train_dir before flow_from_directory: {self.train_dir}") # DEBUG PRINT
        #print(f"val_dir before flow_from_directory: {self.val_dir}") # DEBUG PRINT
        #print(f"test_dir before flow_from_directory: {self.test_dir}") # DEBUG PRINT

        callbacks = [CallbackOverfitPrevention()]
        
        start_time = time.time()
        print("\033[1;32mTraining started...\033[0m")

        # Define data generator
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

        train_generator = datagen.flow_from_directory(
            train_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        val_generator = datagen.flow_from_directory(
            val_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        model = self.model
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            callbacks=callbacks
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\033[1;32mTraining completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes).\033[0m") # 033 for bold text, 32 for blue text

        self.plot_training_history(history)
        
        return history


    def test(self, num_classes, dataset_path):

        train_dir, val_dir, test_dir = None, None, None
        
        if not self.split_sets:
            if os.path.exists(self.split_dataset_path): # Check if split directory exists
                print("\033[1;34mLoading dataset...\033[0m")
                train_dir = os.path.join(self.split_dataset_path, 'train')
                val_dir = os.path.join(self.split_dataset_path, 'val')
                test_dir = os.path.join(self.split_dataset_path, 'test')
            else:
               train_dir, val_dir, test_dir = self.split_dataset() # Perform dataset splitting
            self.split_sets = True # Set the flag
       
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

        train_dir = self.train_dir
        val_dir = self.val_dir
        test_dir = self.test_dir

        #print(f"train_dir before flow_from_directory: {self.train_dir}") # DEBUG PRINT
        #print(f"val_dir before flow_from_directory: {self.val_dir}") # DEBUG PRINT
        #print(f"test_dir before flow_from_directory: {self.test_dir}") # DEBUG PRINT

        start_time = time.time()
        print("\033[1;32mEvaluation started...\033[0m")


        # Define data generator
        datagen = ImageDataGenerator(
             preprocessing_function=preprocess_input
        )

        test_generator = datagen.flow_from_directory(
            test_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        predictions = self.model.predict(test_generator)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\033[1;32mEvaluation completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes).\033[0m")

        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes

        print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

        self.plot_confusion_matrix(y_true, y_pred, test_generator.class_indices.keys())


    # Plot training & validation accuracy values
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()
    

    # Plot confusion matrix
    def plot_confusion_matrix(self, y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        dataset_folder_name = os.path.basename(self.dataset_path)
        if dataset_folder_name == 'Combined':
            plt.figure(figsize=(50, 50)) # Bigger size for the 71 classes
        else:
            plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Reds', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
     

    def save(self, model_save_path):
        full_save_path = model_save_path + '_' + str(BATCH_SIZE) + '-' + str(EPOCHS) + '.weights.h5'
        self.model.save_weights(full_save_path, overwrite = True)


    def load(model_save_path, num_classes, dataset_path):
        dataset_folder_name = os.path.basename(dataset_path)
        if dataset_folder_name == 'BSL(1000)' or dataset_folder_name == 'ASL(1000)':
            LEARNING_RATE = 1e-5
        else:
            LEARNING_RATE = 1e-4
        model=ResNet(num_classes, dataset_path)
        full_save_path = model_save_path + '_' + str(BATCH_SIZE) + '-' + str(EPOCHS) + '.weights.h5'
        model.model.load_weights(full_save_path, skip_mismatch=False)
        model.model.compile(keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss = 'categorical_crossentropy', metrics =['accuracy'])
        return model