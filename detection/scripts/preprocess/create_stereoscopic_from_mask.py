# Note: You most likely do not need this file. It was to create a stereoscopic dataset from which I would run the other scripts

import os
from PIL import Image
import numpy as np
import random
import argparse

from detection_and_crop.utils.create_stereoscopic_dataset_utils import square_crop_with_padding


def parse_args():
    parser = argparse.ArgumentParser(description='Crop images and labels based on masks.')
    parser.add_argument('--image_input_folder', type=str, help='Path to the input image folder. So your color fundus photos .png')
    parser.add_argument('--label_input_folder', type=str, help='Path to the input label folder. So your masks, in my case, 3 channel cup disc background masks .png')
    parser.add_argument('--output_image_folder', type=str, help='Path to the output image folder.')
    parser.add_argument('--output_label_folder', type=str, help='Path to the output label folder.')
    parser.add_argument('--crop_min', type=float, default=0.125, help='Mean for random shift noise (default: 0).')
    parser.add_argument('--crop_max', type=float, default=0.175, help='Standard deviation for random shift noise (default: 25)')
    parser.add_argument('--max_offset', type=float, default=0.5, help='Mean for random shift noise (default: 10).')
    return parser.parse_args()


def main():
    """Crops images and labels based on masks, adding optional noise and zoom."""
    args = parse_args()

    # Ensure output folders exist
    os.makedirs(args.output_image_folder, exist_ok=True)
    os.makedirs(args.output_label_folder, exist_ok=True)

    # List PNG files in the input folder
    png_files = [file for file in os.listdir(args.label_input_folder) if file.endswith('.png')]

    for png_file in png_files:
        # Load images and masks
        input_image = Image.open(os.path.join(args.image_input_folder, png_file))
        label_mask = Image.open(os.path.join(args.label_input_folder, png_file))

        max_dim = max(input_image.height, input_image.width)

        # Convert to NumPy arrays
        # input_image_array = np.array(input_image)
        label_mask_array = np.array(label_mask)

        # Find mask coordinates
        # mask_coordinates = np.argwhere(label_mask_array[:, :, 1] > 0)
        mask_coordinates = np.argwhere(label_mask_array > 0)

        # # Calculate bounding box
        min_y, min_x = np.min(mask_coordinates, axis=0)
        max_y, max_x = np.max(mask_coordinates, axis=0)

        padding = int(np.random.uniform(args.crop_min, args.crop_max) * max_dim)
        cropped_image, cropped_label = square_crop_with_padding(input_image, label_mask, (min_x, min_y, max_x, max_y), padding, args.max_offset)
        if cropped_image is not None and cropped_label is not None:
            cropped_image = Image.fromarray(cropped_image)
            cropped_label = Image.fromarray(cropped_label)

            # Save cropped files
            output_image_file = os.path.join(args.output_image_folder, png_file)
            cropped_image.save(output_image_file)
            output_label_file = os.path.join(args.output_label_folder, png_file)
            cropped_label.save(output_label_file)


if __name__ == '__main__':
    main()
