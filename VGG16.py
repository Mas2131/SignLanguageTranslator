import keras, os
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.callbacks import Callback
from preprocess_utils import IMG_SIZE
from generators import generators
#To use a pre-trained Network, saving time 
from keras.applications.vgg16 import VGG16

# An accuracy threshold help preventing overfitting
threshold = 98e-2

class CallbackOverfitPrevention(Callback):
    def prevent_overfitting(self, epoch, logs = None):
        if(logs.ger('accuracy') >= threshold):
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

        self.build(None, IMG_SIZE, IMG_SIZE, 1)

        self.compile(keras.optimizers.Adam(learning_rate= alpha), loss = 'categorical_crossentropy', metrics =['accuracy'] )
        self.summary()
"""

#With pretrained Network
class VGGNet(keras.Model):
    def __init__(self, n_letters):
        super(VGGNet, self).__init__()
        self.shape = (IMG_SIZE, IMG_SIZE, 3)
        self.letters = n_letters
        self.model = None

        alpha = 1e-4 #Learning rate
        
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
        self.model.compile(keras.optimizers.Adam(learning_rate= alpha), loss = 'categorical_crossentropy', metrics =['accuracy'] )
        
    def training(self, train_data, train_labels):
        # Debug
        model = self.model 
        model.summary()

        train_set, validation_set = generators(train_data, train_labels)

        fitted_model = model.fit(
            train_set,
            steps_per_epoch = 4, #Reduced epocs dor testing if the dataset is working
            epochs = 5, #Reduced epocs dor testing if the dataset is working
            validation_data = validation_set,
            validation_steps= 2,    #Reduced epocs dor testing if the dataset is working
            callbacks= [CallbackOverfitPrevention()]
        )

        return fitted_model
    
    # Save the weights of the model
    def save(self, file_path):
        self.save_weights(file_path, overwrite = True)

    #Loads the weights of the model for evaluation
    def load(model, file_path):
        model.load_weights(file_path, skip_mismatch=False)
        return model
        