# Train YOLO Model

## Usage

```bash
python train_yolo.py \
    --data path/to/coco_yaml_.yaml \
    --model path/to/pretrained/model.pt \
    --epochs 200 \
    --imgsz 640 \
    --lr0 0.000005 \
    --patience 10
```