"""
build_intel_quality_dataset.py

This script reads images from the Intel Image Classification dataset,
computes simple "quality" metrics (brightness and contrast),
and saves them into a CSV file called 'intel_camera_quality.csv'.

You run this first, then use analyze_intel_quality.py on the CSV it creates.
"""

import os              # For working with folders and file paths
import numpy as np     # For numeric array operations
import pandas as pd    # For building and saving the table (DataFrame)
from PIL import Image  # From the Pillow library, to open and process images


# -----------------------------
# CONFIGURATION SECTION
# -----------------------------

# Path to the folder containing the training images.
# We expect a structure like:
#   seg_train/seg_train/buildings
#   seg_train/seg_train/forest
#   ... etc.
#
# If your folder structure is different, you can change this path.
TRAIN_DIR = os.path.join("seg_train", "seg_train")

# Number of images to process per class (category).
# The Intel dataset is big; to keep it fast, we limit how many we read.
IMAGES_PER_CLASS = 100  # You can reduce to 50 if your machine is slow


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def compute_brightness_and_contrast(image_path):
    """
    Open an image, convert it to grayscale, and compute:
      - brightness: the mean pixel value
      - contrast:  the standard deviation of pixel values

    Parameters
    ----------
    image_path : str
        The path to the image file.

    Returns
    -------
    (brightness, contrast) : (float, float) or (None, None)
        If successful, returns brightness and contrast.
        If an error occurs, returns (None, None).
    """
    try:
        # Open the image file
        img = Image.open(image_path)

        # Convert the image to grayscale ("L" mode = 8-bit pixels, black and white)
        img = img.convert("L")

        # Convert the grayscale image to a NumPy array for numeric operations
        arr = np.array(img, dtype=np.float32)

        # Brightness is the average of all pixel values
        brightness = arr.mean()

        # Contrast is the standard deviation of pixel values
        contrast = arr.std()

        return brightness, contrast

    except Exception as e:
        # If something goes wrong (corrupt file, etc.), print error and skip
        print(f"Error processing {image_path}: {e}")
        return None, None


# -----------------------------
# MAIN SCRIPT LOGIC
# -----------------------------

def main():
    """
    Main function that:
      1. Checks that the training directory exists
      2. Loops over each class (buildings, forest, etc.)
      3. Reads up to IMAGES_PER_CLASS images per class
      4. Computes brightness and contrast for each image
      5. Stores all results in a DataFrame
      6. Saves the DataFrame to 'intel_camera_quality.csv'
    """

    # Check if the training directory actually exists
    if not os.path.isdir(TRAIN_DIR):
        print(f"ERROR: Training directory not found → {TRAIN_DIR}")
        print("Make sure your folder structure is correct.")
        return

    # This list will collect one dictionary per image (one row per image)
    rows = []

    # This will be a numeric ID we assign to each image (0, 1, 2, 3, ...)
    image_id = 0

    # Loop over each subfolder inside TRAIN_DIR
    # Each subfolder is expected to be a label like "buildings", "forest", etc.
    for label in sorted(os.listdir(TRAIN_DIR)):
        # Build the full path to the class folder
        class_dir = os.path.join(TRAIN_DIR, label)

        # If it's not a folder (e.g., a file by mistake), skip it
        if not os.path.isdir(class_dir):
            continue

        print(f"Processing class: {label}")

        # We'll use 'count' to ensure we only process up to IMAGES_PER_CLASS
        count = 0

        # Loop over files in the class directory
        for filename in os.listdir(class_dir):
            # If we've already processed enough images for this class, stop
            if count >= IMAGES_PER_CLASS:
                break

            # Only process image files with these extensions
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # Build the full file path to the image
            image_path = os.path.join(class_dir, filename)

            # Compute brightness and contrast for this image
            brightness, contrast = compute_brightness_and_contrast(image_path)

            # If we failed to compute, skip this image
            if brightness is None:
                continue

            # Build a dictionary representing one row of data
            row = {
                "Image_ID": image_id,     # Our own internal ID
                "Filepath": image_path,   # Where the file is located
                "Label": label,           # The class (buildings, forest, etc.)
                "Brightness": brightness, # Computed brightness
                "Contrast": contrast      # Computed contrast
            }

            # Append this row to our list
            rows.append(row)

            # Increment counters
            image_id += 1
            count += 1

    # Convert the list of rows into a pandas DataFrame (table)
    df = pd.DataFrame(rows)

    # Name of the CSV file we will create
    output_csv = "intel_camera_quality.csv"

    # Save the DataFrame to CSV in the current folder
    df.to_csv(output_csv, index=False)

    print(f"Finished! Saved {len(df)} rows to {output_csv}")


# This ensures main() runs when we call "python build_intel_quality_dataset.py"
if __name__ == "__main__":
    main()
