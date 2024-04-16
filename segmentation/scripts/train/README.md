# Train Segmentation Model

Currently using Mask2Former

## Usage

```bash

python train_mask2former.py \
    --train_csv /path/to/train.csv \
    --val_csv /path/to/val.csv \
    --csv_img_path_col image_path \
    --csv_label_path_col label_path \
    --output_directory /path/to/outputs \
    --dataset_mean 0.768 0.476 0.290 \
    --dataset_std 0.220 0.198 0.166 \
    --lr 0.00002 \
    --batch_size 8 \
    --jitters 0.5 0.5 0.25 0.1 0.75 \
    --num_epochs 200 \
    --patience 10 \
    --num_val_outputs_to_save 7 \
    --num_workers 16

```