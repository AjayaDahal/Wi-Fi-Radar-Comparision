import os
import cv2
import h5py
import numpy as np

# Define the class labels
class_labels = {
    'Fall': 0,
    'LieDown': 1,
    'Pickup': 2,
    'Run': 3,
    'Sitdown': 4,
    'Standup': 5,
    'Walk': 6,
    'fall': 0,
    'liedown': 1,
    'Liedown': 1,
    'walk': 6
}

# Function to resize and convert an image to grayscale
def resize_and_convert_to_grayscale(image):
    resized_image = cv2.resize(image, (128, 128))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

# Function to process a folder and create an HDF5 dataset
def process_folder(folder_path, hf):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                image_path = os.path.join(root, file)
                # Extract the class and index from the filename
                parts = file.split("_")
                class_name = parts[2]
                index = int(parts[3].split(".")[0])

                if class_name in class_labels:
                    class_label = class_labels[class_name]
                    image = cv2.imread(image_path)
                    grayscale_image = resize_and_convert_to_grayscale(image)

                    if index == 8 or index == 10 or index == 12 or index == 14 :
                        x_test.append(grayscale_image)
                        y_test.append(class_label)
                                                #print("train dataset")
                    else:
                        x_train.append(grayscale_image)
                        y_train.append(class_label)
                        #print("test dataset")

    # Add data to the HDF5 file
    for x, y in zip(x_train, y_train):
        hf["x_train"].resize((hf["x_train"].shape[0] + 1, 128, 128))
        hf["x_train"][-1] = x
        hf["y_train"].resize((hf["y_train"].shape[0] + 1,))
        hf["y_train"][-1] = y

    for x, y in zip(x_test, y_test):
        hf["x_test"].resize((hf["x_test"].shape[0] + 1, 128, 128))
        hf["x_test"][-1] = x
        hf["y_test"].resize((hf["y_test"].shape[0] + 1,))
        hf["y_test"][-1] = y

# Define the path to the root folder containing the 700 subfolders
root_folder = ".\\NewDataset90-120\\"

# Define the path to the HDF5 file
hf_file = ".\\NewDataset90-120\\WiFiDataset7classes5Candidates80_20_90-120_8121416.h5"

# Create an HDF5 file for storing the dataset
with h5py.File(hf_file, "w") as hf:
    hf.create_dataset("x_train", (0, 128, 128), dtype=np.uint8, maxshape=(None, 128, 128))
    hf.create_dataset("y_train", (0,), dtype=np.uint8, maxshape=(None,))
    hf.create_dataset("x_test", (0, 128, 128), dtype=np.uint8, maxshape=(None, 128, 128))
    hf.create_dataset("y_test", (0,), dtype=np.uint8, maxshape=(None,))

    # Process each subfolder
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            process_folder(folder_path, hf)
            print("Processing: "+str(folder_path))

print("Dataset creation complete.")
