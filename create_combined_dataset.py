"""
Combines the three datasets with 300 images each. To distinguish between the different alphabets, it adds the appropriate suffix to each letter/class.

Debugging print have been commented out.

Author: Anna Pia Mascolo
"""

import os
import shutil

dest_folder = './Datasets/Unzipped/Combined'

def create_combined_dataset():
    # Ensure the destination folder exists, if not it creates it
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    src_folder_ASL = './Datasets/Unzipped/ASL(300)'
    suffix_ASL = '-ASL'

    for root, dirs, files in os.walk(src_folder_ASL):
        for subdir in dirs:
            # Current folder path
            subdir_path = os.path.join(root, subdir)
            renamed_folder = subdir + suffix_ASL
            new_path = os.path.join(dest_folder, renamed_folder)
            # print(new_path)
            # Copy the folder to the destination with the new name
            shutil.copytree(subdir_path, new_path)

    src_folder_BSL = './Datasets/Unzipped/BSL(300)'
    suffix_BSL = '-BSL'

    for root, dirs, files in os.walk(src_folder_BSL):
        for subdir in dirs:
            # Current folder path
            subdir_path = os.path.join(root, subdir)
            renamed_folder = subdir + suffix_BSL
            new_path = os.path.join(dest_folder, renamed_folder)
            # print(new_path)
            # Copy the folder to the destination with the new name
            shutil.copytree(subdir_path, new_path)


    src_folder_LIS = './Datasets/Unzipped/LIS(300)'
    suffix_LIS = '-LIS'

    for root, dirs, files in os.walk(src_folder_LIS):
        for subdir in dirs:
            # Current folder path
            subdir_path = os.path.join(root, subdir)
            renamed_folder = subdir + suffix_LIS
            new_path = os.path.join(dest_folder, renamed_folder)
            # print(new_path)
            # Copy the folder to the destination with the new name
            shutil.copytree(subdir_path, new_path)


