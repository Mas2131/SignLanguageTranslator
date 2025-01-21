"""
    Resizes .png images by the given scale and converts them to .jpg format.

    Parameters:
        input_root_folder (str): the path to the folder containing subfolders with .png images.
        output_root_folder (str): the path to the folder where to save the subfolders with the resized .jpg images.
        scale (float): the scale factor to resize images (set to 0.25 for 75% reduction).

    
    
    Author: Ida Perfetto
"""

import os
from PIL import Image

def resize_and_convert_images(input_root_folder, output_root_folder, resize_factor=0.25):
    print("Script started")
    print(f"Input folder content: {os.listdir(input_root_folder)}")
    
    for folder_name in os.listdir(input_root_folder):
        folder_path = os.path.join(input_root_folder, folder_name)
        print(f"Checking folder: {folder_path}")
        
        if os.path.isdir(folder_path):
            output_folder = os.path.join(output_root_folder, folder_name)
            print(f"Creating output folder: {output_folder}")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            for filename in os.listdir(folder_path):
                print(f"Processing file: {filename}")
                if filename.endswith('.png'):
                    input_path = os.path.join(folder_path, filename)
                    output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.jpg")
                    print(f"Resizing and converting {filename}")
                    
                    try:
                        with Image.open(input_path) as img:
                            new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
                            img = img.resize(new_size, Image.ANTIALIAS)
                            img = img.convert("RGB")
                            img.save(output_path, 'JPEG', quality=85)
                            print(f"Saved {output_path}")
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

input_root_folder = '...'
output_root_folder = '...'
resize_and_convert_images(input_root_folder, output_root_folder)

