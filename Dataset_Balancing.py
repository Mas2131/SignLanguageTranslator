"""
This script is designed to balance the class distribution of a dataset of images organized into subfolders (classes)

It aims to increase or reduce the number of images in each subfolder to reach a predefined target_image_count: 
- for under-represented classes, new images are created as variations from existing ones with image augmentation techniques
- for over-represented classes, random images are deleted

Authors: Ida Perfetto & Anna Pia Mascolo

"""

import os
import random
import numpy as np
from PIL import Image
import shutil
import imgaug.augmenters as iaa

# Define the path to the base folder containing the original image dataset
base_folder = './LIS-fingerspelling-dataset'
# Define the path to the output folder where augmented images will be stored.
output_folder = './LIS-fingerspelling-dataset(300)'
# Defines the desired number of images per subfolder
target_image_count=300


# Define the augmentation sequence using imgaug
augmenter = iaa.Sequential([
    iaa.Affine(rotate=(-30, 30), shear=(-20, 20)),
    iaa.Fliplr(0.5),
    iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="constant", pad_cval=255),
    iaa.AddToBrightness((-30, 30)),
    iaa.AddToHueAndSaturation((-20, 20)),
    iaa.ScaleX((0.8, 1.2)),
    iaa.ScaleY((0.8, 1.2))
])
# augmenter applies:
# random rotation between -30 and 30 degrees
# shear between -20 and 20 degrees
# horizontal flip with a 50% probability
# cropping and padding with a percentage between -20% and 20% with white pixels(255)
# brightness with a random value between -30 and 30
# hue and saturation with a random value between -30 and 30



# List all subfolders in the base folder
subfolders = [os.path.join(base_folder, subfolder) for subfolder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, subfolder))]

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for subfolder in subfolders:

    subfolder_name = os.path.basename(subfolder)
    # List all image files in the current subfolder
    images = [f for f in os.listdir(subfolder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_count = len(images)

    print(f"\033[1;34mProcessing folder: {subfolder_name}\033[0m")
    print(f"Current image count: {image_count}")    

    # Create a corresponding subfolder in the output folder
    output_subfolder = os.path.join(output_folder, os.path.basename(subfolder))
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    # Copy original images to the new folder first
    for image in images:
        original_image_path = os.path.join(subfolder, image)
        destination_image_path = os.path.join(output_subfolder, image)
        shutil.copy(original_image_path, destination_image_path)
    

    if image_count < target_image_count:
        # Calculate how many images to augment
        augment_needed = target_image_count - image_count
        print(f"Augment needed: {augment_needed}")
        augmented_count = 0

        while augmented_count < augment_needed:
            # Randomly select an image for augmentation
            image = random.choice(images)
            image_path = os.path.join(subfolder, image)

            # Load the image using PIL
            img = Image.open(image_path)
            img_array = np.array(img)

            # Augment the image
            augmented_images = augmenter(images=[img_array])

            for aug_img in augmented_images:
                # Save the augmented image
                aug_img = Image.fromarray(aug_img)
                aug_img.save(os.path.join(output_subfolder, f"aug_{image.split('.')[0]}_{augmented_count}.jpeg"))
                augmented_count += 1

                if augmented_count >= augment_needed:
                    break

        # Re-check total image count to ensure correctness
        final_image_count = len([f for f in os.listdir(output_subfolder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if final_image_count != target_image_count:
            print(f"Error: Expected {target_image_count} images but found {final_image_count} in {output_subfolder}.")
        else:
            print(f"\033[1;32mSuccessfully added {augment_needed} images in {subfolder_name} to reach exactly {target_image_count} images.\033[0m")


    elif image_count > target_image_count:

        
        # Calculate how many images to delete
        reduction_needed = image_count - target_image_count
        print(f"Reduction needed: {reduction_needed}")
        
        
        # Get a list of all files in the new folder
        files = [os.path.join(output_subfolder, f) for f in os.listdir(output_subfolder) if os.path.isfile(os.path.join(output_subfolder, f))]
       
        deleted_count = 0
        
        while deleted_count < reduction_needed:
            files_to_delete = random.sample(files, reduction_needed)
            
            for file in files_to_delete:
                os.remove(file)
                print(f"Deleted: {file}")
                deleted_count += 1

                if deleted_count >= reduction_needed:
                    break
                
            # Update the list of files to avoid trying to delete already removed files 
            files = [f for f in files if f not in files_to_delete]

        
        # Re-check total image count to ensure correctness
        final_image_count = len([f for f in os.listdir(output_subfolder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if final_image_count != target_image_count:
            print(f"Error: Expected {target_image_count} images but found {final_image_count} in {output_subfolder}.")
        else:
            print(f"\033[1;31mSuccessfully deleted {reduction_needed} images in {subfolder_name} to reach exactly {target_image_count} images\033[0m")