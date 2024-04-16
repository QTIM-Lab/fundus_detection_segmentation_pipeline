# COCO Dataset

A COCO dataset is needed to train and validate the YOLO model. Users may provide their own. Otherwise, this module will create a COCO dataset for you, based on the segmentation labels (i.e. by setting bounding box to be extent of segmentation region)

## Scripts to create the COCO dataset

### Create COCO .txt files

```bash
python create_coco_dataset.py \
    --input_folder /path/to/dataset/labels_images \
    --output_folder /path/to/dataset/labels_txt \
    --padding 0
```

### Create Train, Val Splits

Can change for own needs

#### Usage

```bash
python create_coco_splits.py \
    --images_path /path/to/dataset/images \
    --labels_path /path/to/dataset/labels_txt \
    --output_path /path/to/output \
    --split_ratio 0.85
```