import argparse
import os
import random
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Split a dataset into train and validation sets.')
    parser.add_argument('--images_path', type=str, help='Path to the input images folder')
    parser.add_argument('--labels_path', type=str, help='Path to the input labels folder')
    parser.add_argument('--output_path', type=str, help='Path to the output directory')
    parser.add_argument('--split_ratio', type=float, default=0.85, help='Split ratio for training data (default: 0.85)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Input
    images_path = args.images_path
    labels_path = args.labels_path

    # Output
    output_path = args.output_path
    split_ratio = args.split_ratio

    # Create output directories
    os.makedirs(os.path.join(output_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'val'), exist_ok=True)

    # Get the list of image files
    image_files = os.listdir(images_path)

    # Shuffle the list to ensure randomness in the split
    random.shuffle(image_files)

    # Calculate the split index
    split_index = int(len(image_files) * split_ratio)

    # Split the images and labels into train and validation sets
    train_images = image_files[:split_index]
    val_images = image_files[split_index:]

    # Copy the images and labels to the respective directories
    for image_file in train_images:
        image_path = os.path.join(images_path, image_file)
        label_path = os.path.join(labels_path, image_file.split('.')[0] + '.txt')

        shutil.copy(image_path, os.path.join(output_path, 'images', 'train', image_file))
        shutil.copy(label_path, os.path.join(output_path, 'labels', 'train', image_file.split('.')[0] + '.txt'))

    for image_file in val_images:
        image_path = os.path.join(images_path, image_file)
        label_path = os.path.join(labels_path, image_file.split('.')[0] + '.txt')

        shutil.copy(image_path, os.path.join(output_path, 'images', 'val', image_file))
        shutil.copy(label_path, os.path.join(output_path, 'labels', 'val', image_file.split('.')[0] + '.txt'))


if __name__ == '__main__':
    main()
