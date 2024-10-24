# Note: You most likely do not need this file. It was to create a stereoscopic dataset from which I would run the other scripts

import os
from PIL import Image
import numpy as np
import pandas as pd
import argparse

from detection.utils.create_stereoscopic_dataset_utils import square_crop_with_padding_no_label


def parse_args():
    parser = argparse.ArgumentParser(description='Crop images and labels based on masks.')
    parser.add_argument('--input_folder', type=str, help='Path to the input image folder. So your color fundus photos .png')
    parser.add_argument('--output_folder', type=str, help='Path to the output image folder.')
    parser.add_argument('--crop_min', type=float, default=0.125, help='Mean for random shift noise (default: 0).')
    parser.add_argument('--crop_max', type=float, default=0.175, help='Standard deviation for random shift noise (default: 25)')
    parser.add_argument('--max_offset', type=float, default=0.5, help='Mean for random shift noise (default: 10).')
    return parser.parse_args()


def main():
    """Crops images and labels based on masks, adding optional noise and zoom."""
    args = parse_args()

    # Ensure output folders exist
    os.makedirs(args.output_image_folder, exist_ok=True)

    # List PNG files in the input folder
    # png_files = [file for file in os.listdir(args.label_input_folder) if file.endswith('.png')]
    # Load data from CSV using pandas
    csv_data = pd.read_csv(args.csv_file)

    for index, row in csv_data.iterrows():
        filename = row["filename"]
        x, y, w, h = map(int, [row["x"], row["y"], row["w"], row["h"]])

        # Load images and masks
        input_image = Image.open(os.path.join(args.image_input_folder, filename))

        max_dim = max(input_image.height, input_image.width)

        # Calculate bounding box
        min_x, min_y, max_x, max_y = x, y, x + w, y + h

        padding = int(np.random.uniform(args.crop_min, args.crop_max) * max_dim)
        cropped_image = square_crop_with_padding_no_label(input_image, (min_x, min_y, max_x, max_y), padding, args.max_offset)
        if cropped_image is not None:
            cropped_image = Image.fromarray(cropped_image)

            # Save cropped files
            output_image_file = os.path.join(args.output_image_folder, filename)
            cropped_image.save(output_image_file)


if __name__ == '__main__':
    main()
