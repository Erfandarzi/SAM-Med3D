import os

def rename_images_in_directory(directory_path, old_suffix="_0000.nii.gz", new_suffix=".nii.gz"):
    """
    Renames image files in the specified directory by replacing the old_suffix with new_suffix.
    """
    if not os.path.exists(directory_path):
        print(f"Directory does not exist: {directory_path}")
        return
    
    for filename in os.listdir(directory_path):
        if filename.endswith(old_suffix):
            new_name = filename.replace(old_suffix, new_suffix)
            os.rename(os.path.join(directory_path, filename), os.path.join(directory_path, new_name))
            print(f"Renamed {filename} to {new_name}")

# List of task names to process
task_names = [
    "Task506_NWMH",
    "Task508_ISLES2022",
    "Task507_JHU",
    "Task505_ISLES",
    "Task504_ATLAS",
    "Task503_INRIA",
    "Task502_BCHUNC",
    "Task501_HIE"
]

# Base directories
base_train_dir = "./data/train/brain_lesion/"
base_validation_dir = "./data/validation/brain_lesion/"

# Process each task
for task_name in task_names:
    images_dir = os.path.join(base_train_dir, task_name, "imagesTr")
    images_test_dir = os.path.join(base_validation_dir, task_name, "imagesTs")
    
    # Rename image files in training and validation directories
    rename_images_in_directory(images_dir)
    rename_images_in_directory(images_test_dir)


