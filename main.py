from Dataset import load_dataset
from VGG16 import *
import argparse, os

import tensorflow as tf

#My PC is not cuda compatible, therefore I disabled the GPU and the related warning
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#For LIS
n_letters = 22
FILE_PATH = 'VGG_LIS_model.keras'
DATASET_PATH = './archive/LIS-fingerspelling-dataset/'
CLASSES = ["a", "b", "c", "d", "e", "f", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "t", "u", "v", "w", "x" "y"]


#For BSL
#n_letters = 10
#FILE_PATH = 'VGG_BSL_model.keras'

#For ASL
#n_letters = 10
#FILE_PATH = 'VGG_ASL_model.keras'


# To prevent retraining the model every time the code is run, 
# testing and training functions have been separated and can be called while running the code
# This has been learned during the Reinforcement Learning course

def test():
    model = VGGNet.load(FILE_PATH, n_letters)
    _, test_data, _, test_labels = load_dataset(DATASET_PATH)
    eval_met = model.test(test_data, test_labels, CLASSES)
    # Print the evaluation results
    print("Evaluation results are:")
    print("Loss: ", eval_met[0])
    print("Accuracy: ", eval_met[1])
    print("False Positives: ", eval_met[2])
    print("False Negatives: ", eval_met[3])
    print("True Positives: ", eval_met[4])
    print("True Negatives: ", eval_met[5])
    print("Recall: ", eval_met[6])
    print("Precision: ", eval_met[7])


def train():
    model = VGGNet(n_letters)
    train_data, _, train_label, _ = load_dataset(DATASET_PATH)
    model.training(train_data, train_label, FILE_PATH)
    model.save(FILE_PATH)
    print("Model has been correctly saved.")

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