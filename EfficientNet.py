""" 
This class implements the EfficientNetV2 architecture using a pretrained network.

Author: Viviana Serra 
"""

#Necessary imports
import numpy as np
import pandas as pd
import os
import pathlib
import matplotlib.pyplot as plt
import shutil
import seaborn as sns
import tensorflow as tf
import time
import psutil
import keras
from sklearn.model_selecstion import train_test_split
from keras.applications import EfficientNetV2B0
from keras.applications.efficientnet_v2 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, classification_report

# General parameters
img_size = (224, 224)
batch_size = 2
epochs = 1
THRESHOLD = 0.98
# Callbacks for save and early stopping
checkpoint = ModelCheckpoint("best_mobilenet_v2.keras", save_best_only=True, monitor="val_accuracy", mode="max")
early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)


# Early stopping callback to prevent overfitting
class CallbackOverfitPrevention(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy", 0) >= THRESHOLD or logs.get("val_accuracy", 0) >= THRESHOLD:
            print("\033[1;31mModel is likely overfitted. Stopping training.\033[0m")
            self.model.stop_training = True


def prepare_data_splits(base_path, save_path, train_split=0.6, val_split=0.3):
    """
    Split in test,validation and train
    """
    # Remove the directory is it already exists
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    # Create the directories
    for split in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(save_path, split), exist_ok=True)

    all_images = []  # List of all images
    all_labels = []  # List of corresponding labels

    # Iterate on each class of the dataset to construct global list
    for class_name in os.listdir(base_path):
        if class_name != "_split":
            class_path = os.path.join(base_path, class_name)
            if not os.path.isdir(class_path):
                continue

            images = os.listdir(class_path)
            all_images.extend([os.path.join(class_path, img) for img in images])
            all_labels.extend([class_name] * len(images))

    # Divide the dataset but keep it balanced
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        all_images, all_labels, train_size=train_split, random_state=42, stratify=all_labels
    )
    val_split_adjusted = val_split / (1 - train_split)
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, train_size=val_split_adjusted, random_state=42, stratify=temp_labels
    )

    # Copy in the correspective directories
    for split, images, labels in zip(
        ['train', 'validation', 'test'],
        [train_images, val_images, test_images],
        [train_labels, val_labels, test_labels]
    ):
        for img_path, label in zip(images, labels):
            class_dir = os.path.join(save_path, split, label)
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy(img_path, os.path.join(class_dir, os.path.basename(img_path)))

    # Print to check values
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(save_path, split)
        print(f"Splitting {split}:")
        for class_name in os.listdir(split_path):
            class_split_path = os.path.join(split_path, class_name)
            print(f"  Class {class_name}: {len(os.listdir(class_split_path))} images")



def preprocess_image(image, label):
    """
    Preprocess images scaling pixels values between 0 and 1
    """
    image = tf.image.resize(image, img_size)  # Resize image
    image = preprocess_input(image)
    image = tf.image.resize(image, (224, 224)) / 255.0  # Normalize images
    return image, label

def create_dataset_from_directory(directory, shuffle=True):
    """
    Trasforms images in tf.data.Dataset from directory
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)



class EfficientNet(keras.Model):
    def __init__(self, num_classes, dataset_path, split_dataset_path):
        super(EfficientNet, self).__init__()
        self.num_classes = num_classes
        self.dataset_path = dataset_path
        self.split_dataset_path = split_dataset_path
        self.model = None
        self.build_model()

    def build_model(self):        
        # Upload the pre-trained model EfficientNetV2
        base_model = EfficientNetV2B0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        
        ## Freeze the weights of the model
        base_model.trainable = False
        
        # Add classifier
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001), 
            loss = 'categorical_crossentropy', 
            metrics =['accuracy']
        )

        
    def train(self, file_path):
        # To estimate time of training
        start_time = time.time()
        start_cpu = psutil.cpu_times()
        
        prepare_data_splits(self.dataset_path, self.split_dataset_path)

        # Load of the dataset
        base_train_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=os.path.join(self.split_dataset_path, 'train'),
            image_size=img_size,
            batch_size=batch_size,
            label_mode='categorical'
        )

        # Apply transformation to the dataset
        train_dataset = create_dataset_from_directory(os.path.join(self.split_dataset_path, 'train'))
        val_dataset = create_dataset_from_directory(os.path.join(self.split_dataset_path, 'validation'))
        
        # Print info
        print(f"Training set size: {len(train_dataset)} batches")
        print(f"Validation set size: {len(val_dataset)} batches")

        # Display of random images
        for images, labels in train_dataset.take(1):
            for i in range(5):  # Shows 5 examples
                plt.imshow(images[i].numpy())
                plt.title(f"Label: {labels[i].numpy().argmax()}")
                plt.show()

        model = self.model  

        # Determine the weight for each class
        # Extract labels from training    
        labels = np.concatenate([y.numpy() for _, y in base_train_dataset], axis=0)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(self.num_classes),
            y=labels.argmax(axis=1)  # Convert one-hot to index
        )
        class_weights = dict(enumerate(class_weights))
        
        history = model.fit(
            train_dataset, #training dataset
            epochs=epochs,
            validation_data = val_dataset, # Validation dataset
            callbacks= [checkpoint, early_stopping, CallbackOverfitPrevention]
        )
        
        # End time of computations
        end_time = time.time()
        end_cpu = psutil.cpu_times()

        # Compute time
        wall_time = end_time - start_time
        cpu_time_user = end_cpu.user - start_cpu.user
        cpu_time_system = end_cpu.system - start_cpu.system
        cpu_time_total = cpu_time_user + cpu_time_system

        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours}h {minutes}min {seconds}s"
        
        # Print results
        print(f"CPU times: user {format_time(cpu_time_user)}, sys: {format_time(cpu_time_system)}, total: {format_time(cpu_time_total)}")
        print(f"Wall time: {format_time(wall_time)}")

        model.save(file_path)

        return history
    
    def test(self, class_names):
        # To estimate time of testing
        start_time = time.time()
        start_cpu = psutil.cpu_times()

        # Apply transformation to the dataset
        test_dataset = create_dataset_from_directory(os.path.join(self.split_dataset_path, 'test'))
        
        # Print info
        print(f"Test set size: {len(test_dataset)} batches")
        
        test_loss, test_accuracy = self.model.evaluate(test_dataset)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # End time of computations
        end_time = time.time()
        end_cpu = psutil.cpu_times()

        # Compute time
        wall_time = end_time - start_time
        cpu_time_user = end_cpu.user - start_cpu.user
        cpu_time_system = end_cpu.system - start_cpu.system
        cpu_time_total = cpu_time_user + cpu_time_system

        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours}h {minutes}min {seconds}s"
        
        # Print results
        print(f"CPU times: user {format_time(cpu_time_user)}, sys: {format_time(cpu_time_system)}, total: {format_time(cpu_time_total)}")
        print(f"Wall time: {format_time(wall_time)}")

        # Extraction of true values and of the predictaed values
        y_true = []
        y_pred_probs = []

        # Final prevision
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_true, axis=1)  # Convert one-hot index

        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=class_names)
        print(report)

        # Create confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Normalize confusion matrix
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # Display normalized confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix_normalized, 
                    annot=True, 
                    fmt=".2f", 
                    cmap="Blues",
                    xticklabels=class_names, 
                    yticklabels=class_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Normalized Confusion Matrix")
        plt.show()


    # Save the weights of the model
    def save(self, file_path):
        full_save_path = file_path + '_' + str(batch_size) + '-' + str(epochs) + '.weights.h5'
        self.model.save_weights(full_save_path, overwrite = True)

    #Loads the weights of the model for evaluation
    def load(file_path, n_letters):
        full_save_path = file_path + '_' + str(batch_size) + '-' + str(epochs) + '.weights.h5'
        model = EfficientNet(n_letters)
        model.model.load_weights(full_save_path, skip_mismatch=False)
        model.model.compile(
            optimizer=Adam(learning_rate=0.0001), 
            loss = 'categorical_crossentropy', 
            metrics =['accuracy']
        )
        return model