from Dataset import load_dataset
from generators import generators
from VGG16 import *
import argparse

#For LIS
#Dataset has been reduced to test if the model is working
n_letters = 10
FILE_PATH = 'LIS_model.keras'

#For BLS
#n_letters = 10
#FILE_PATH = 'LSS_model.h5'

#For ASL
#n_letters = 10
#FILE_PATH = 'LIS_model.h5'


# To prevent retraining the model every time the code is run, 
# testing and training functions have been separated and can be called while running the code
# This has been learned during the Reinforcement Learning course

def test():
    model = VGGNet.load(FILE_PATH, n_letters)
    _, test_data, _, test_labels = load_dataset()
    print('Test subset: ')
    loss, accuracy = model.model.evaluate(test_data, test_labels)
    print("Loss: ", loss, "Accuracy: ", accuracy)


def train():
    model = VGGNet(n_letters)
    train_data, _, train_label, _ = load_dataset()
    model.training(train_data, train_label)
    model.save(FILE_PATH)

#Prints for debugging
def main():
    print("Starting dataset loading...")
    """
    try:
        train_data, test_data, train_label, test_label = load_dataset()
        train_generator, validation_generator =  generators(train_data, train_label)
        print("Dataset loaded successfully!")
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        print(f"Train labels shape: {train_label.shape}")
        print(f"Test labels shape: {test_label.shape}")
    except Exception as e:
        print(f"An error occurred: {e}")
    """
    parser = argparse.ArgumentParser(description='Run training and testing')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()

    if args.train:
        train()

    if args.evaluate:
        test()

# Run the main function
if __name__ == "__main__":
    main()