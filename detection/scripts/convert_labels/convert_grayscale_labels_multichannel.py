import argparse
from PIL import Image
import numpy as np
import os

from detection_and_crop.utils.convert_grayscale_utils import convert_to_rgb


def parse_args():
    parser = argparse.ArgumentParser(description='Convert grayscale images to RGB with specific channel assignments.')
    parser.add_argument('--input_folder', help='Path to the folder containing grayscale images')
    parser.add_argument('--output_folder', help='Path to the folder where RGB images will be saved')
    return parser.parse_args()


def main():
    args = parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    
    # Make sure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    input_files = os.listdir(input_folder)

    for file_name in input_files:
        input_path = os.path.join(input_folder, file_name)

        if os.path.isfile(input_path):
            convert_to_rgb(input_path, output_folder)


if __name__ == "__main__":
    main()
