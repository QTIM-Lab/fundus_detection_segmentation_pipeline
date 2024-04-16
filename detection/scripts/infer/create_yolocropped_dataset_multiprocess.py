from ultralytics import YOLO
from PIL import Image
import os
import torch
import pandas as pd
from multiprocessing import Pool, set_start_method, Manager
import argparse
import numpy as np

from detection_and_crop.datasets.yolo_inference_dataset import YoloInferenceDataset
from detection_and_crop.utils.create_yolocropped_utils import process_result

def parse_args():
    parser = argparse.ArgumentParser(description="Process images using YOLO model.")
    parser.add_argument(
        "--root_directory", type=str, help="Path to the root directory containing images."
    )
    parser.add_argument(
        "--output_directory", type=str, help="Path to the output directory for images."
    )
    parser.add_argument(
        "--label_root_directory", type=str, default=None, help="Path to the root directory containing labels for images. Must be same filename as image itself."
    )
    parser.add_argument(
        "--label_output_directory", type=str, default=None, help="Path to the output directory for segmentation labels."
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to the YOLO model checkpoint file."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Confidence threshold for bounding box selection. Between 0 and 1. Use the visualize yolo script to see some values that work best, note the conf value",
    )
    parser.add_argument(
        "--output_img_size",
        type=int,
        default=512,
        help="Size to resize cropped images to. I.e. what size does your segmentation model take. My segmentation model also takes (512,512) both MaskFormer and SegFormer both did",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Path to the YOLO model checkpoint file."
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=16,
        help="Number of parallel processes to use. Note this is CPU multiprocessing, handling batch outputs",
    )
    return parser.parse_args()
    

def main():
    args = parse_args()

    set_start_method('spawn')
    # mp.set_start_method("spawn", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make sure output directories exist
    os.makedirs(args.output_directory, exist_ok=True)
    csv_output_dir = os.path.join(args.output_directory, 'csvs')
    os.makedirs(csv_output_dir, exist_ok=True)
    if args.label_output_directory is not None:
        os.makedirs(args.label_output_directory, exist_ok=True)

    # Define model
    model = YOLO(args.model_path).to(device)

    # Define images that will be used
    img_files = [
        file for file in os.listdir(args.root_directory) if file.lower().endswith((".png", ".jpg"))
    ]

    assert len(img_files) > 0, "Did not detect any images. Make sure they end with .png or .jpg (case insensitive)"

    # Example usage:
    dataset = YoloInferenceDataset(args.root_directory, args.batch_size, label_directory=args.label_root_directory)

    batch_num = 0
    pool = Pool(processes=args.num_processes)

    # Collect results from the queue
    results = []
    
    # To manage queue
    with Manager() as manager:
        # Create a multiprocessing queue to store processed rows
        pool_queue = manager.Queue()

        for batch, batch_files, batch_labels in dataset:
            batch_num += 1

            res = model(batch)
            
            for i, result_i in enumerate(res):
                file_name = batch_files[i]
                img = batch[i]
                label_img = None
                if len(batch_labels) > 0:
                    label_img = batch_labels[i]
                pool.apply_async(process_result, args=(pool_queue, result_i, img, file_name, args.output_directory, batch_num, i, args.threshold, args.output_img_size, 25, label_img, args.label_output_directory))
    
        pool.close()
        pool.join()

        while not pool_queue.empty():
            row = pool_queue.get()
            results.append(row)

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Write DataFrame to CSV
    df.to_csv(os.path.join(csv_output_dir, 'bbox_output.csv'), index=False)

    print("Inference on all images completed. CSV written")


if __name__ == "__main__":
    main()
