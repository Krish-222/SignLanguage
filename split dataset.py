import os
import shutil

# Paths
source_folder = "self_generated_dataset"
train_folder = "train_dataset"
test_folder = "test_dataset"

# Create train and test folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Loop through subfolders
for subfolder_name in os.listdir(source_folder):
    subfolder_path = os.path.join(source_folder, subfolder_name)
    
    if os.path.isdir(subfolder_path):
        images = os.listdir(subfolder_path)
        num_images = len(images)
        
        train_count = min(num_images, 400)  # Number of images to put in the train folder
        
        # Create subfolders in train and test folders
        train_subfolder = os.path.join(train_folder, subfolder_name)
        test_subfolder = os.path.join(test_folder, subfolder_name)
        
        os.makedirs(train_subfolder, exist_ok=True)
        os.makedirs(test_subfolder, exist_ok=True)
        
        # Move images to appropriate folders
        for i, image_name in enumerate(images):
            source_image_path = os.path.join(subfolder_path, image_name)
            if i < train_count:
                target_image_path = os.path.join(train_subfolder, image_name)
            else:
                target_image_path = os.path.join(test_subfolder, image_name)
                
            shutil.move(source_image_path, target_image_path)

print("Dataset split completed.")
