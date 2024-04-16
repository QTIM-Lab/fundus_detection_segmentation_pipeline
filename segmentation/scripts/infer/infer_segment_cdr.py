import argparse
import os
import torch
from multiprocessing import Pool, set_start_method, Manager
import numpy as np
from transformers import MaskFormerImageProcessor, Mask2FormerForUniversalSegmentation
import albumentations as A
from torch.utils.data import DataLoader

from segmentation_train_and_inference.datasets.infer_datasets import SegmentationInferenceDataset
from segmentation_train_and_inference.utils.segmentation_utils import color_palette
from segmentation_train_and_inference.utils.infer_utils import process_segmentation, get_data_rows, write_output_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Train MaskFormer model for instance segmentation")
    parser.add_argument("--model_path", type=str, default='/path/to/weights.pt', help="Path to trained model weights")
    parser.add_argument("--input_csv", type=str, default='/path/for/images.csv', help="Path to csv to run inference on all rows")
    parser.add_argument("--input_root_dir", type=str, default=None, help="Path to images to run inference on all rows. If none, use the actual csv path")
    parser.add_argument("--csv_path_col_name", required=True, default='img_path', help="Column name for the paths to images in the csv, ex: file_path or img_path")
    parser.add_argument("--csv_dataset_col_name", default=None, help="Column name for dataset if provided, to fix naming conventions")
    parser.add_argument("--csv_label_col_name", default=None, help="Column name for labels if provided, to calculate CDR for them as well")
    parser.add_argument("--output_root_dir", type=str, required=True, default='./', help="Path to outputs for CSV and images. CSV will be saved to root dir, images to root_dir/outputs")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of threads to run in parallel")
    parser.add_argument('--dataset_mean', nargs='+', type=float, help='Array of float values for mean i.e. 0.709 0.439 0.287')
    parser.add_argument('--dataset_std', nargs='+', type=float, help='Array of float values for std i.e. 0.210 0.220 0.199')
    parser.add_argument("--num_processes", type=int, default=1, help="Number of threads to run in parallel")
    parser.add_argument("--cuda_num", type=int, default=0, help="Cuda device to run on")
    
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path
    input_csv = args.input_csv
    input_root_dir = args.input_root_dir
    csv_path_col_name = args.csv_path_col_name
    csv_dataset_col_name = args.csv_dataset_col_name
    csv_label_col_name = args.csv_label_col_name
    output_root_dir = args.output_root_dir
    num_processes = args.num_processes
    cuda_num = args.cuda_num
    batch_size = args.batch_size
    dataset_mean = args.dataset_mean
    dataset_std = args.dataset_std

    # Create the output dir for the images
    output_dir = os.path.join(output_root_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    # Specify output file path to be hard-coded as results.csv in the output dir
    output_csv_file = os.path.join(output_root_dir, f'results.csv')
    
    # Multiprocess set to spawn
    set_start_method('spawn')

    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print('using device: ', device)

    # for classes
    id2label = {
        0: "unlabeled",
        1: "background",
        2: "disc",
        3: "cup"
    }

    # for vis
    palette = color_palette()

    # transforms. use what you calculated for train/val
    ADE_MEAN = np.array(dataset_mean)
    ADE_STD = np.array(dataset_std)

    test_transform = A.Compose([
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)
    ])

    # Create an instance of your custom dataset
    dataset = SegmentationInferenceDataset(input_csv, input_root_dir, csv_path_col_name, transform=test_transform, label_col_name=csv_label_col_name)

    # Create a data loader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_processes // 2)


    # Replace the head of the pre-trained model
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic",
                                                            id2label=id2label,
                                                            ignore_mismatched_sizes=True).to(device)

    # Load the state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create a preprocessor
    preprocessor = MaskFormerImageProcessor(ignore_index=0, do_reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

    # Create mp pool
    pool = Pool(processes=num_processes // 2)

    # To manage queue
    with Manager() as manager:
        # Create a multiprocessing queue to store processed rows
        result_queue = manager.Queue()

        # Iterate through the dataset
        for batch_data, image_paths, label_segs, idxs in dataloader:

            # prepare image for the model
            inputs = preprocessor(images=batch_data, return_tensors="pt").to(device)

            with torch.no_grad():
                # Process the batch using your model
                outputs = model(**inputs)

            # Output is always (512,512) for all images in batch
            target_sizes = [(512, 512) for _ in range(len(batch_data))]

            # Huggingface get the seg maps. Wanted to do this in parallel...
            predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                        target_sizes=target_sizes)

            # For each of the outputs on the batch, process it
            for i, predicted_seg_i in enumerate(predicted_segmentation_maps):
                # Get segs, filename, idx
                predicted_seg_i_np = predicted_seg_i.cpu().numpy()
                file_name = os.path.basename(image_paths[i])

                label_seg = None if label_segs[i] == '' else label_segs[i].cpu().numpy()
                idx_num = idxs[i].item()

                # Apply MP function
                pool.apply_async(process_segmentation, args=(predicted_seg_i_np, file_name, result_queue, palette, idx_num, output_root_dir, label_seg))
                # process_segmentation(predicted_seg_i_np, file_name, result_queue, palette, idx_num, output_root_dir, label_seg)
        
        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()

        # Collect data rows
        all_rows = get_data_rows(result_queue)

    write_output_csv(output_csv_file, all_rows)

    print("Inference on all images completed.")


if __name__ == '__main__':
    main()
