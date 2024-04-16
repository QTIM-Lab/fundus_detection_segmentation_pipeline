# Preprocess

## Create Stereoscopic Dataset

### From detect coordinates

```bash

python create_stereoscopic_dataset_from_detect.py \
    --input_folder /path/to/input \
    --output_folder /path/to/output \
    --crop_min 0.15 \
    --crop_max 0.15 \
    --max_offset 0.0

```