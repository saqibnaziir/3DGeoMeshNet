import os
from PIL import Image
import numpy as np

def crop_center_region(input_folder, output_folder):
    """
    Crop the center region of the images (human face) by removing white surrounding shapes.
    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save cropped images.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path).convert("RGB")  # Convert to RGB
            
            # Convert image to NumPy array for processing
            img_np = np.array(img)
            
            # Create a mask for non-white pixels
            mask = np.any(img_np < [240, 240, 240], axis=-1)
            
            # Find coordinates of the bounding box of the face (non-white region)
            coords = np.argwhere(mask)
            if coords.size == 0:
                print(f"Skipping {filename}: No face detected.")
                continue
            
            y0, x0 = coords.min(axis=0)  # Top-left corner
            y1, x1 = coords.max(axis=0)  # Bottom-right corner
            
            # Crop the image using the bounding box
            cropped_img = img.crop((x0, y0, x1 + 1, y1 + 1))
            
            # Save the cropped image to the output folder
            output_path = os.path.join(output_folder, filename)
            cropped_img.save(output_path)
            print(f"Cropped and saved: {output_path}")

# Input and output folder paths
input_folder = "/home/nazir241/PosDoc/COSURAI/Codes/FaceCom/Denoising/Figs/gt"   # Replace with the folder containing your images
output_folder = "/home/nazir241/PosDoc/COSURAI/Codes/FaceCom/Denoising/Figs/gt_croped" # Replace with the folder to save cropped images

# Run the cropping function
crop_center_region(input_folder, output_folder)

