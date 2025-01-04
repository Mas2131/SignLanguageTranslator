import keras
import numpy as np

# To create network from scratch
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.callbacks import Callback
from preprocess_utils import IMG_SIZE
from generators import generators
#To use a pre-trained Network, saving time 
from keras.applications.vgg16 import VGG16

# To evaluate the model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# An accuracy threshold help preventing overfitting
threshold = 98e-2
BATCH_SIZE = 16 #32 #64
EPOCHS_SIZE = 10 #15 #20
alpha = 1e-4 #Learning rate


# To prevent overfitting, the training is stopped when it reaches 98% of accuracy
class CallbackOverfitPrevention(Callback):
    def prevent_overfitting(self, epoch, logs = None):
        if(logs.get('accuracy') >= threshold):
            print("Model is probably overfittes. Training has stopped")
            self.model.stop_training = True



"""
#Builg from sctratch
class CNNNetwork(keras.Model):
    #Since 2 letters in the LIS alphabet require movement,
    #we will consider only 22 letters
    def __init__(self, n_letters):
        super().__init__()

        #Some hyperparameters
        alpha = 1e-4 #Learning rate
        self = Sequential()

        #Network
        #First block
        self.conv11 = Conv2D(64, (3, 3), activation='relu', padding = 'same', input_shape=(IMG_SIZE, IMG_SIZE, 1))
        self.conv12 = Conv2D(64, (3, 3), activation='relu', padding = 'same')
        self.pool1 = MaxPool2D((2, 2))

        #Second block
        self.conv21 = Conv2D(128, (3, 3), activation='relu', padding = 'same')
        self.conv22 = Conv2D(128, (3, 3), activation='relu',  padding = 'same')
        self.pool2 = MaxPool2D((2, 2))

        #Third block
        self.conv31 = Conv2D(256, (3, 3), activation='relu', padding = 'same')
        self.conv32 = Conv2D(256, (3, 3), activation='relu', padding = 'same')
        self.conv33 = Conv2D(256, (3, 3), activation='relu', padding = 'same')
        self.pool3 = MaxPool2D((2, 2))

        #Fourth block
        self.conv41 = Conv2D(512, (3, 3), activation='relu', padding = 'same')
        self.conv42 = Conv2D(512, (3, 3), activation='relu', padding = 'same')
        self.conv43 = Conv2D(512, (3, 3), activation='relu', padding = 'same')
        self.pool4 = MaxPool2D((2, 2))

        #Fifth block
        self.conv51 = Conv2D(512, (3, 3), activation='relu', padding = 'same')
        self.conv52 = Conv2D(512, (3, 3), activation='relu', padding = 'same')
        self.conv53 = Conv2D(512, (3, 3), activation='relu', padding = 'same')
        self.pool5 = MaxPool2D((2, 2))

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(4096, activation='relu')
        self.dropout1 = Dropout(0.5)
        self.dense2 = Dense(4096, activation='relu')
        self.dropout2 = Dropout(0.5)
        self.dense3 = Dense(n_letters, activation='softmax')

        self.build(None, IMG_SIZE, IMG_SIZE, 3)

        self.model.compile(keras.optimizers.Adam(learning_rate= alpha), loss = 'categorical_crossentropy', 
                           metrics =[keras.metrics.Accuracy(),
                                     keras.metrics.FalsePositives(),
                                     keras.metrics.FalseNegatives(),
                                     keras.metrics.TruePositives(),
                                     keras.metrics.TrueNegatives(),
                                     keras.metrics.Recall(),
                                     keras.metrics.Precision()] )
        
        self.summary()
"""

# With pretrained Network
class VGGNet(keras.Model):
    def __init__(self, n_letters):
        super(VGGNet, self).__init__()
        self.shape = (IMG_SIZE, IMG_SIZE, 3)
        self.letters = n_letters
        self.model = None
        self.build_model()

    def build_model(self):        
        # weights='imagenet' fetches the pretrained hyperparameters
        vgg16 = VGG16(weights='imagenet', input_shape=self.shape, classes= self.letters, include_top= False)
        
        # Freezes the weights so that are not retrained, saving time
        for layer in vgg16.layers:
            layer.trainable = False
        
        # When working with a pretrained model, we only need to change the last layer:
        x = Flatten()(vgg16.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.letters, activation='softmax')(x)

        self.model = Model(inputs=vgg16.input, outputs=predictions)
        self.model.compile(keras.optimizers.Adam(learning_rate= alpha), loss = 'categorical_crossentropy', 
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

        train_set, validation_set = generators(train_data, train_labels)

        fitted_model = model.fit(
            train_set,
            batch_size= BATCH_SIZE,
            epochs = EPOCHS_SIZE, 
            validation_data = validation_set,
            callbacks= [CallbackOverfitPrevention()]
        )
        
        model.save(file_path)

        history = fitted_model.history

        # Plot training & validation accuracy
        plt.figure(figsize=(12, 6))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Show plots
        plt.tight_layout()
        plt.show()
        return fitted_model
    
    def plot_confusion_matrix(self, y_true, y_pred, classes):
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, normalize="true")

        fig, ax = plt.subplots(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.YlGn, ax=ax, values_format=".2f")

        # Adjust font size for better readibility
        plt.setp(ax.get_xticklabels(), fontsize=10)  
        plt.setp(ax.get_yticklabels(), fontsize=10)  
        for labels in ax.texts:  
            labels.set_fontsize(8)
        plt.title("Normalized Confusion Matrix", fontsize = 16)

        # To prevent overlapping
        plt.tight_layout()
        plt.show()

        # Save cf image
        plt.savefig('Confusion_matrix_VGG.png')  
        plt.close()
        
    
    def test(self, test_data, test_labels, classes):
        results = self.model.evaluate(test_data, test_labels, batch_size= BATCH_SIZE)
        #print("Evaluation results:", results)
        y_prediction = self.model.predict(test_data)
        # Debugging
        #print("y_true:", test_labels)
        #print("classes:", classes)
        #print("y_pred:", y_prediction)
        self.plot_confusion_matrix(test_labels, y_prediction, classes)
        return results


    # Save the weights of the model
    def save(self, file_path):
        #self.model.save(file_path, overwrite = True) working but using format .keras gives problems during the evaluation
        self.model.save_weights(file_path, overwrite = True)

    #Loads the weights of the model for evaluation
    def load(file_path, n_letters):
        model = VGGNet(n_letters)
        model.model.load_weights(file_path, skip_mismatch=False)
        model.model.compile(keras.optimizers.Adam(learning_rate= alpha), loss = 'categorical_crossentropy', 
                           metrics =['accuracy',
                                     keras.metrics.FalsePositives(),
                                     keras.metrics.FalseNegatives(),
                                     keras.metrics.TruePositives(),
                                     keras.metrics.TrueNegatives(),
                                     keras.metrics.Recall(),
                                     keras.metrics.Precision()] )
        
        return model