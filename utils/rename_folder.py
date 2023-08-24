import os

def rename_folders(directory):
    for folder_name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, folder_name)):
            new_name = folder_name.split("_align_corners")[0]
            os.rename(os.path.join(directory, folder_name), os.path.join(directory, new_name))
            print(f"Renamed {folder_name} to {new_name}")

# Specify the directory path where the folders are located
folder_directory = 'LightningTrainer_2023-07-08_12-41-40'

# Call the function to rename folders
rename_folders(folder_directory)

