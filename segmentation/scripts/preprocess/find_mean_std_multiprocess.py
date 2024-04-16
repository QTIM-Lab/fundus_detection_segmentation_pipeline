import argparse
import pandas as pd

from segmentation_train_and_inference.utils.preprocess_utils import calculate_mean_std


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate mean and standard deviation of images.")
    parser.add_argument("--csv", required=True, help="CSV file containing file paths")
    parser.add_argument("--csv_path_col_name", required=True, help="Column name for the paths in the csv, ex: file_path or img_path")
    parser.add_argument("--img_size", type=int, default=224, help="Size of the images (assume square)")
    parser.add_argument("--num_processes", type=int, default=None, help="Number of processes to use (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    image_paths = df[args.csv_path_col_name].tolist()

    print("Number of images:", len(image_paths))

    mean, std = calculate_mean_std(image_paths, image_size=(args.img_size, args.img_size), num_processes=args.num_processes)
    print("Mean:", mean)
    print("Standard Deviation:", std)


if __name__ == "__main__":
    main()
