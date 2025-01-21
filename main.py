"""
This script serves as the main entry point for training and evaluating the proposed models on the different proposed datasets

The supported models are:
- ResNet50
- VGG16
- VGG19
- MobileNetV2
- EfficientNet

The expected datasets are:
- ASL(1000)
- ASL(300)
- BSL(1000)
- BSL(300)
- LIS(1000)
- LIS(300)
- Combined

Features:
- Automatically unzips datasets
- Allows users to choose from the 5 proposed models and 7 proposed datasets through command-line arguments
- Enables training and evaluation through command-line arguments

Usage:
- Train a model:
  python main.py -t -m resnet -d lis1000

- Evaluate a model:
  python main.py -e -m resnet -d lis1000

replace resnet with the argument for the desired model, lis1000 with the desired dataset

"""

import argparse 
from ResNet50 import *
#from VGG16 import *
#from VGG16 import load_dataset as load_16
#from VGG19 import *
#from VGG19 import load_dataset as load_19
#from MobileNet import *
#from EfficientNet import *
import os
import zipfile 
import shutil

# Unzip the dataset from the zip file
def unzip_dataset(zip_path, extract_path):
    print("\033[1;31mDataset unzipping started...\033[0m") # 033 for bold text, 31 for red text
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            print(f"\033[1;31mDataset uzipping completed. You can find the unzipped dataset in: {extract_path}\033[0m")
    except FileNotFoundError:
        print(f"Error: Zip file not found at {zip_path}. Please check the path.")
    except Exception as e:
        print(f"Error during unzipping: {e}")

# Get the list of class folders in the dataset directory
def get_classes(dataset_path):
    classes = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
    classes.sort() # Sort the classes alphabetically 
    return classes
    

def main():

    # Define base paths for datasets and model weights
    DATA_BASE_PATH = './Datasets/' # Define the base directory in where the zip files are stored
    UNZIPPED_DATA_PATH = './Datasets/Unzipped' # Define the directory where the extracted datasets will be placed
    WEIGHTS_PATH = './Weights' # Define the directory where the weights of the models will be saved
    os.makedirs(UNZIPPED_DATA_PATH, exist_ok=True) # Create the unzipped dataset directory if it doesn't exist, if it exist do nothing
    os.makedirs(WEIGHTS_PATH, exist_ok=True) # Create the model weights directory if it doesn't exist, if it exist do nothing


    parser = argparse.ArgumentParser(description="Train or evaluate the classifier on the BSL, LIS, or ASL dataset.")
    parser.add_argument('-t', '--train', action='store_true', help='Train the model') # Train flag
    parser.add_argument('-e', '--evaluate', action='store_true', help='Evaluate the model') # Evaluate flag
    parser.add_argument('-m', '--model', type=str, choices=['resnet', 'vgg16', 'vgg19', 'mobilenet', 'efficientnet'], required=True, help='Choose the model to use: ResNet50, VGG16, VGG19, MobileNetV2 or EfficientNet') # Model choice
    parser.add_argument('-d', '--dataset', type=str, choices=['asl300', 'asl1000', 'bsl300', 'bsl1000', 'lis300', 'lis1000', 'combined'], required=True, help='Choose the dataset to use: bsl, lis, or asl') # Dataset choice
    args = parser.parse_args()

    
    # Dataset-specific configuration for BSL datasets
    if args.dataset == 'bsl1000' or args.dataset == 'bsl300':
        if args.dataset == 'bsl1000':
            zip_path = os.path.join(DATA_BASE_PATH, 'BSL(1000).zip') # Create the full path of the zip file for the BSL dataset with 1000 images per class
            dataset_path = os.path.join(UNZIPPED_DATA_PATH, 'BSL(1000)') # Create the full path for the unzipped BSL dataset with 1000 images per class
            if args.model == 'resnet':
                model_save_path = os.path.join(WEIGHTS_PATH, 'ResNet50_BSL-1000_model')
            elif args.model == 'vgg16':
                model_save_path = os.path.join(WEIGHTS_PATH, 'VGG16_BSL-1000_model')
            elif args.model == 'vgg19':
                model_save_path = os.path.join(WEIGHTS_PATH, 'VGG19_BSL-1000_model')
            elif args.model == 'mobilenet':
                model_save_path = os.path.join(WEIGHTS_PATH, 'MobileNet_BSL-1000_model')
                split_dataset_save_path = os.path.join(dataset_path, '_split')
            else:
                model_save_path = os.path.join(WEIGHTS_PATH, 'EfficientNet-1000_BSL_model')
                split_dataset_save_path = os.path.join(dataset_path, '_split')

        else:
            zip_path = os.path.join(DATA_BASE_PATH, 'BSL(300).zip') # Create the full path of the zip file for the BSL dataset with 300 images per class
            dataset_path = os.path.join(UNZIPPED_DATA_PATH, 'BSL(300)') # Create the full path for the unzipped BSL dataset with 300 images per class
            if args.model == 'resnet':
                model_save_path = os.path.join(WEIGHTS_PATH, 'ResNet50_BSL-300_model')
            elif args.model == 'vgg16':
                model_save_path = os.path.join(WEIGHTS_PATH, 'VGG16_BSL-300_model')
            elif args.model == 'vgg19':
                model_save_path = os.path.join(WEIGHTS_PATH, 'VGG19_BSL-300_model')
            elif args.model == 'mobilenet':
                model_save_path = os.path.join(WEIGHTS_PATH, 'MobileNet_BSL-300_model')
                split_dataset_save_path = os.path.join(dataset_path, '_split')
            else:
                model_save_path = os.path.join(WEIGHTS_PATH, 'EfficientNet_BSL-300_model')
                split_dataset_save_path = os.path.join(dataset_path, '_split')
        num_classes = 23
    

    # Dataset-specific configuration for LIS datasets
    elif args.dataset == 'lis300' or args.dataset == 'lis1000':
        if args.dataset == 'lis300':
            zip_path = os.path.join(DATA_BASE_PATH, 'LIS(300).zip') # Create the full path of the zip file for the LIS dataset with 300 images per class
            dataset_path = os.path.join(UNZIPPED_DATA_PATH, 'LIS(300)') # Create the full path for the unzipped LIS dataset with 300 images per class
            if args.model == 'resnet':
                model_save_path = os.path.join(WEIGHTS_PATH, 'ResNet50_LIS-300_model')
            elif args.model == 'vgg16':
                model_save_path = os.path.join(WEIGHTS_PATH, 'VGG16_LIS-300_model')
            elif args.model == 'vgg19':
                model_save_path = os.path.join(WEIGHTS_PATH, 'VGG19_LIS-300_model')
            elif args.model == 'mobilenet':
                model_save_path = os.path.join(WEIGHTS_PATH, 'MobileNet_LIS-300_model')
                split_dataset_save_path = os.path.join(dataset_path, '_split')
            else:
                model_save_path = os.path.join(WEIGHTS_PATH, 'EfficientNet_LIS-300_model')
                split_dataset_save_path = os.path.join(dataset_path, '_split')
        else:
            zip_path = os.path.join(DATA_BASE_PATH, 'LIS(1000).zip') # Create the full path of the zip file for the LIS dataset with 1000 images per class
            dataset_path = os.path.join(UNZIPPED_DATA_PATH, 'LIS(1000)') # Create the full path for the unzipped LIS dataset with 1000 images per class
            if args.model == 'resnet':
                model_save_path = os.path.join(WEIGHTS_PATH, 'ResNet50_LIS-1000_model')
            elif args.model == 'vgg16':
                model_save_path = os.path.join(WEIGHTS_PATH, 'VGG16_LIS-1000_model')
            elif args.model == 'vgg19':
                model_save_path = os.path.join(WEIGHTS_PATH, 'VGG19_LIS-1000_model')
            elif args.model == 'mobilenet':
                model_save_path = os.path.join(WEIGHTS_PATH, 'MobileNet_LIS-1000_model')
                split_dataset_save_path = os.path.join(dataset_path, '_split')
            else:
                model_save_path = os.path.join(WEIGHTS_PATH, 'EfficientNet_LIS-1000_model')
                split_dataset_save_path = os.path.join(dataset_path, '_split')
        num_classes = 22
    

    # Dataset-specific configuration for ASL datasets
    elif args.dataset == 'asl1000' or args.dataset == 'asl300':
        if args.dataset == 'asl1000':
            zip_path = os.path.join(DATA_BASE_PATH, 'ASL(1000).zip') # Create the full path of the zip file for the ASL dataset with 1000 images per class
            dataset_path = os.path.join(UNZIPPED_DATA_PATH, 'ASL(1000)') # Create the full path for the unzipped ASL dataset with 1000 images per class
            if args.model == 'resnet':
                model_save_path = os.path.join(WEIGHTS_PATH, 'ResNet50_ASL-1000_model')
            elif args.model == 'vgg16':
                model_save_path = os.path.join(WEIGHTS_PATH, 'VGG16_ASL-1000_model')
            elif args.model == 'vgg19':
                model_save_path = os.path.join(WEIGHTS_PATH, 'VGG19_ASL-1000_model')
            elif args.model == 'mobilenet':
                model_save_path = os.path.join(WEIGHTS_PATH, 'MobileNet_ASL-1000_model')
                split_dataset_save_path = os.path.join(dataset_path, '_split')
            else:
                model_save_path = os.path.join(WEIGHTS_PATH, 'EfficientNet_ASL-1000_model')
                split_dataset_save_path = os.path.join(dataset_path, '_split')

        else:
            zip_path = os.path.join(DATA_BASE_PATH, 'ASL(300).zip') # Create the full path of the zip file for the ASL dataset with 300 images per class
            dataset_path = os.path.join(UNZIPPED_DATA_PATH, 'ASL(300)') # Create the full path for the unzipped ASL dataset with 300 images per class
            if args.model == 'resnet':
                model_save_path = os.path.join(WEIGHTS_PATH, 'ResNet50_ASL-300_model')
            elif args.model == 'vgg16':
                model_save_path = os.path.join(WEIGHTS_PATH, 'VGG16_ASL-300_model')
            elif args.model == 'vgg19':
                model_save_path = os.path.join(WEIGHTS_PATH, 'VGG19_ASL-300_model')
            elif args.model == 'mobilenet':
                model_save_path = os.path.join(WEIGHTS_PATH, 'MobileNet_ASL-300_model')
                split_dataset_save_path = os.path.join(dataset_path, '_split')
            else:
                model_save_path = os.path.join(WEIGHTS_PATH, 'EfficientNet_ASL-300_model')
                split_dataset_save_path = os.path.join(dataset_path, '_split')
        num_classes = 26
    
    # Dataset-specific configuration for Combined datasets
    elif args.dataset == 'combined':
        zip_path = os.path.join(DATA_BASE_PATH, 'Combined.zip') # Create the full path of the zip file for the Combined dataset
        dataset_path = os.path.join(UNZIPPED_DATA_PATH, 'Combined') # Create the full path for the unzipped Combined dataset
        if args.model == 'resnet':
            model_save_path = os.path.join(WEIGHTS_PATH, 'ResNet50_Combined_model')
        elif args.model == 'vgg16':
            model_save_path = os.path.join(WEIGHTS_PATH, 'VGG16_Combined_model')
        elif args.model == 'vgg19':
            model_save_path = os.path.join(WEIGHTS_PATH, 'VGG19_Combined_model')
        elif args.model == 'mobilenet':
            model_save_path = os.path.join(WEIGHTS_PATH, 'MobileNet_Combined_model')
            split_dataset_save_path = os.path.join(dataset_path, '_split')
        else:
            model_save_path = os.path.join(WEIGHTS_PATH, 'EfficientNet_Combined_model')
            split_dataset_save_path = os.path.join(dataset_path, '_split')
        num_classes = 71

    # Handle invalid dataset argument
    else:
        print("Invalid dataset selected, pease specify either -d asl1000, -d asl300, -d bsl1000, -d bsl300, -d lis1000, -d lis300 or -d combined.")
        return

    # Check if dataset is already unzipped, otherwise unzip
    if not os.path.exists(dataset_path):
        unzip_dataset(zip_path, UNZIPPED_DATA_PATH)
            

    if args.model == 'vgg16' or args.model == 'vgg19':
        if args.model == 'vgg16':
            print(f"Model selected: VGG16")
        else:
            print(f"Model selected: VGG19")
        classes = get_classes(dataset_path) 
        print(f"Classes: {classes}")
    
    elif args.model == 'resnet':
        print(f"Model selected: ResNet50")

    elif args.model == 'efficientnet':
        print(f"Model selected: EfficientNet")
    
    elif args.model == 'mobilenet':
        print(f"Model selected: MobileNet")
        
    else:
        print("Invalid model selected, pease specify either -m resnet, -m vgg16, -m vgg19, -m mobilenet or -m efficientnet.")
        return


    # Training logic
    if args.train:
        if args.model == 'resnet':
            model = ResNet(num_classes, dataset_path)
            model.train(model_save_path)
            model.save(model_save_path)
        
        elif args.model == 'vgg16' or args.model == 'vgg19':
            if args.model == 'vgg16':
                model = VGGNet16(num_classes)
                train_data, _, train_label, _ = load_16(dataset_path)
            else:
                model = VGGNet19(num_classes)
                train_data, _, train_label, _ = load_19(dataset_path)
                
            model.training(train_data, train_label, model_save_path)
            model.save(model_save_path)
        
        elif args.model == 'mobilenet' or args.model == 'efficientnet':
            if args.model == 'mobilenet':
                model = MobileNet(num_classes, dataset_path, split_dataset_save_path)
                model.train(model_save_path)
            else:
                model = EfficientNet(num_classes, dataset_path, split_dataset_save_path)
                model.train(model_save_path)
        
        print("Training has ended. Model has been saved.")


    # Evaluation logic
    elif args.evaluate:
        if args.model == 'resnet':
            model = ResNet.load(model_save_path, num_classes, dataset_path)
            model.test(num_classes, dataset_path)
            
        elif args.model == 'vgg16' or args.model == 'vgg19':
            if args.model == 'vgg16':
                model = VGGNet16.load(model_save_path, num_classes)
            else:
                model = VGGNet19.load(model_save_path, num_classes)

            _, test_data, _, test_labels = load_dataset(dataset_path)
            model.test(test_data, test_labels, classes)
        
        elif args.model == 'mobilenet':
            model = MobileNet.load(model_save_path, num_classes)
            model.test(classes)


        else:
            model = EfficientNet.load(model_save_path, num_classes)
            _, test_data, _, test_labels = load_dataset(dataset_path)
            model.test(test_data, test_labels, classes)

            
        print("Evaluation has ended.")

    else:
        print("Please specify either --train or --evaluate.")

    # Optionally remove unzipped files (after training or evaluation):
    # shutil.rmtree(dataset_path)

if __name__ == "__main__":
    main()
