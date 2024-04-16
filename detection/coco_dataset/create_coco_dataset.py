import argparse
import os
from PIL import Image
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Convert segmentation disc labels from PNG to text files.')
    parser.add_argument('--input_folder', type=str, help='Path to the folder containing the PNG label images.')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where the text files will be saved.')
    parser.add_argument('--padding', type=int, default=0, help='Amount of padding to add around the segmentation area.')
    return parser.parse_args()

def main():
    args = parse_args()

    output_folder = args.output_folder
    input_folder = args.input_folder
    padding = args.padding

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all PNG files in the input folder
    png_files = [file for file in os.listdir(input_folder) if file.endswith('.png')]

    for png_file in png_files:
        # Load the PNG image from the "labels/" folder
        img = Image.open(os.path.join(input_folder, png_file)).convert('L')
        
        # Convert the image to a NumPy array for easier processing
        img_array = np.array(img)
        
        # Treat the green channel as a mask and find the coordinates of non-zero pixels
        # mask_coordinates = np.argwhere(img_array[:, :, 1] > 0)
        # use grayscale
        mask_coordinates = np.argwhere(img_array > 0)
        
        # Calculate the bounding box from the mask coordinates
        min_y, min_x = np.min(mask_coordinates, axis=0)
        max_y, max_x = np.max(mask_coordinates, axis=0)
        
        # Calculate the bounding box dimensions
        x_center = (min_x + max_x) / 2 / img_array.shape[1]
        y_center = (min_y + max_y) / 2 / img_array.shape[0]

        # Clip width and height to stay within the [0, 1] range
        # Calculate the maximum allowed width and height
        max_width = min((1 - x_center) * 2, x_center * 2)
        max_height = min((1 - y_center) * 2, y_center * 2)

        # Ensure that width and height do not exceed the maximum allowed values
        width = min(max((max_x - min_x + padding) / img_array.shape[1], 0), max_width)
        height = min(max((max_y - min_y + padding) / img_array.shape[0], 0), max_height)

        # Define the output file path and name
        txt_file = os.path.splitext(png_file)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_file)
        
        # Save the label values in the .txt file
        with open(txt_path, 'w') as txt_file:
            txt_file.write(f'0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}')


if __name__ == '__main__':
   main()
