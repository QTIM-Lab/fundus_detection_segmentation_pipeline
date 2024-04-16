# Create YOLO Cropped Dataset

## Usage

```bash

python create_yolocropped_dataset_multiprocess.py \
    --root_directory /path/to/input/images \
    --output_directory /path/to/output \
    --model_path /path/to/model \
    --threshold 0.875 \
    --output_img_size 512 \
    --batch_size 16 \
    --num_processes 64

```