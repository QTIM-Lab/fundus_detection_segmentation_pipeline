# Infer with Segmentation Model

## Usage

```bash

python infer_segment_cdr.py \
    --model_path /path/to/segmentation/model.pt \
    --input_root_dir /path/to/images \
    --input_csv /path/to/data.csv \
    --csv_path_col_name image_path \
    --output_root_dir /path/to/output \
    --batch_size 8 \
    --dataset_mean 0.768 0.476 0.290 \
    --dataset_std 0.220 0.198 0.166 \
    --num_processes 16 \
    --cuda_num 0

```