import cv2
import os
import random
import numpy as np


# Define the augmentations to be applied
def random_rotate(image):
    angle = random.randint(-25, 25)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    return cv2.warpAffine(image, matrix, (width, height))


def random_blur(image):
    kernel_size = random.choice([3, 5, 7])
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def random_colorspace(image):
    conversion_code = random.choice([cv2.COLOR_BGR2GRAY, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2YCrCb])
    return cv2.cvtColor(image, conversion_code)


# Define the path to the parent directory containing the input directories
parent_dir = "Dataset Batik"

# Define the name of the output directory
output_dir_name = "augmented_images"

# Loop through each input directory and apply the augmentations to the images
for input_dir_name in os.listdir(parent_dir):
    # Skip any non-directory files in the parent directory
    if not os.path.isdir(os.path.join(parent_dir, input_dir_name)):
        continue

    # Define the input and output directory paths
    input_dir = os.path.join(parent_dir, input_dir_name)
    output_dir = os.path.join(parent_dir, input_dir_name + "_" + output_dir_name)

    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all the image file names in the input directory
    file_names = os.listdir(input_dir)

    # Loop through each image file and apply the augmentations
    for file_name in file_names:
        # Load the original image
        print(f'Augmenting Image of {input_dir}_{file_name}')
        image = cv2.imread(os.path.join(input_dir, file_name))

        # Get the dimensions of the original image
        height, width, _ = image.shape

        # Apply the augmentations
        augmented_images = [random_rotate(image), random_blur(image), random_colorspace(image)]

        # Loop through each augmented image and save it to the output directory
        for i, augmented_image in enumerate(augmented_images):
            # Compute the transformation matrix to shift the augmented image to the same location as the original image
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, 1)
            matrix[0, 2] += (width - augmented_image.shape[1]) / 2
            matrix[1, 2] += (height - augmented_image.shape[0]) / 2

            # Apply the transformation matrix to the augmented image
            augmented_image = cv2.warpAffine(augmented_image, matrix, (width, height))

            # Save the augmented image to the output directory
            output_file_name = f"{file_name.split('.')[0]}_{i}.jpg"
            output_file_path = os.path.join(output_dir, output_file_name)
            cv2.imwrite(output_file_path, augmented_image)

            print(f"Saved augmented image to {output_file_path}")
