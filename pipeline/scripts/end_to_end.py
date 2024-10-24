import argparse
import os
from ultralytics import YOLO
import torch
import numpy as np
import albumentations as A
from transformers import MaskFormerImageProcessor, Mask2FormerForUniversalSegmentation

from segmentation.utils.segmentation_utils import color_palette
from pipeline.utils.pipeline_utils import handle_file

def parse_args():
    parser = argparse.ArgumentParser(description="Train MaskFormer model for instance segmentation")
    parser.add_argument("--detection_model_path", type=str, help="Path to trained detection model weights")
    parser.add_argument("--segmentation_model_path", type=str, help="Path to trained segmentation model weights")
    parser.add_argument("--input_dir", type=str, default=None, help="Path to images to run inference on")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to outputs")
    parser.add_argument('--dataset_mean', nargs='+', type=float, help='Array of float values for mean i.e. 0.709 0.439 0.287')
    parser.add_argument('--dataset_std', nargs='+', type=float, help='Array of float values for std i.e. 0.210 0.220 0.199')
    parser.add_argument("--cuda_num", type=int, default=0, help="Cuda device to run on")
    
    return parser.parse_args()


def main():
    args = parse_args()
    detection_model_path = args.detection_model_path
    segmentation_model_path = args.segmentation_model_path
    input_dir = args.input_dir
    output_dir = args.output_dir
    dataset_mean = args.dataset_mean
    dataset_std = args.dataset_std
    cuda_num = args.cuda_num

    # Make output folder if not exist
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

    # Define model
    yolo_model = YOLO(detection_model_path).to(device)

    transform = A.Compose([
        A.Normalize(mean=tuple(dataset_mean), std=tuple(dataset_std))
    ])
    palette = color_palette()
    id2label = {
        0: "unlabeled",
        1: "bg",
        2: "disc",
        3: "cup"
    }
    # Replace the head of the pre-trained model
    seg_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic",
                                                                id2label=id2label,
                                                                ignore_mismatched_sizes=True).to(device)
    # Load the state dictionary
    seg_model.load_state_dict(torch.load(segmentation_model_path, map_location=device))
    seg_model.eval()

    # Create a preprocessor
    preprocessor = MaskFormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

    # Check if the folder exists
    if os.path.exists(input_dir):
        # Loop over files in the folder
        for filename in os.listdir(input_dir):
            print(filename)
            input_file = os.path.join(input_dir, filename)
            handle_file(input_file, yolo_model, seg_model, transform, preprocessor, palette, device, output_dir)
    else:
        print(f"The folder '{input_dir}' does not exist.")


if __name__ == '__main__':
    main()
