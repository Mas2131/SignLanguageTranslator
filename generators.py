from keras.preprocessing.image import ImageDataGenerator

def generators(train_images, train_labels, batch_size=8):
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