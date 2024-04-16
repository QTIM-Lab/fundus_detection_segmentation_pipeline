from PIL import Image
import numpy as np
import os


def convert_to_rgb(input_path, output_dir, bg_thresh=50, cup_thresh=205):
    '''
    Converts a grayscale image to rgb and saves it
    '''
    # Load grayscale image
    grayscale_img = Image.open(input_path).convert('L')

    # Convert to numpy array for easier manipulation
    img_array = np.array(grayscale_img)

    # Create RGB image with distinct channels
    rgb_img = np.zeros((*img_array.shape, 3), dtype=np.uint8)

    # Set red channel for background
    red_condition = (img_array < bg_thresh) 
    rgb_img[:, :, 0] = red_condition * 255  # Set red channel where grayscale is 0

    # Set green channel for specific range
    green_condition = (img_array >= bg_thresh) & (img_array <= cup_thresh)
    rgb_img[:, :, 1] = green_condition * 255  # Set green channel for the specified range

    # Set blue channel for specific range
    blue_condition = (img_array > cup_thresh)
    rgb_img[:, :, 2] = blue_condition * 255  # Set blue channel for the specified range

    # Construct the output path using the filename from the input_path and the output_dir
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    # Convert numpy array back to PIL Image
    final_img = Image.fromarray(rgb_img)

    # Save the result to the constructed output path
    final_img.save(output_path)
