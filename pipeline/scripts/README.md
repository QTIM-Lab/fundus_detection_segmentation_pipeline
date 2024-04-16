# Pipeline end-to-end

Crop and recover segmentation onto a fundus image

Note: Fundus images must have their black edges removed by cropping to largest area with a threshold of 5

## Usage

```bash

python pipeline/scripts/end_to_end.py \
    --detection_model_path /path/to/detect/model.pt \
    --segmentation_model_path /path/to/seg/model.pt \
    --input_dir /path/to/input/images \
    --output_dir /path/to/output/folder \
    --dataset_mean 0.768 0.476 0.290 \
    --dataset_std 0.220 0.198 0.166 \
    --cuda_num 0

```