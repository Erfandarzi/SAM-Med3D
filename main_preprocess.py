import os

task_name="Task507_JHU"
images_dir = "./data/train/brain_lesion/"+task_name+"/imagesTr"
images_test_dir = "./data/train/brain_lesion/"+task_name+"/imagesTs"

# Rename image files
for filename in os.listdir(images_dir):
    if filename.endswith("_0000.nii.gz"):
        new_name = filename.replace("_0000", "")
        os.rename(os.path.join(images_dir, filename), os.path.join(images_dir, new_name))

for filename in os.listdir(images_test_dir):
    if filename.endswith("_0000.nii.gz"):
        new_name = filename.replace("_0000", "")
        os.rename(os.path.join(images_test_dir, filename), os.path.join(images_test_dir, new_name))


# # Check for corresponding label files
# for filename in os.listdir(images_dir):
#     label_filename = filename  # Assuming label files have the same name
#     if not os.path.exists(os.path.join(labels_dir, label_filename)):
#         print(f"Label file missing for {filename}")
