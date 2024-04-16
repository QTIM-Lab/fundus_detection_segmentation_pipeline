# Preprocess

## Create Stereoscopic Dataset

### From detect coordinates

```bash
python create_stereoscopic_dataset_from_mask.py \
    --image_input_folder /path/to/input \
    --label_input_folder /path/to/labels \
    --output_image_folder /path/to/output \
    --output_label_folder /path/to/labels/output \
    --crop_min 0.15 \
    --crop_max 0.15 \
    --max_offset 0.0

python create_stereoscopic_dataset_from_detect.py \
    --input_folder /path/to/input \
    --output_folder /path/to/output \
    --crop_min 0.15 \
    --crop_max 0.15 \
    --max_offset 0.0
```
