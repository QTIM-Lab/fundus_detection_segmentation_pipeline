import argparse
from ultralytics import YOLO


def parse_args():
   parser = argparse.ArgumentParser(description='Train a YOLO model.')
   parser.add_argument('--model', type=str, required=True, help='Path to pretrained model file.')
   parser.add_argument('--data', type=str, required=True, help='Path to data config file (.yaml).')
   parser.add_argument('--imgsz', type=int, default=640, help='Input image size.')
   parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
   parser.add_argument('--device', type=int, default=1, help='Device.')
   return parser.parse_args()


def main():
   args = parse_args()

   model = YOLO(args.model)  # Load pretrained model

   model.val(data=args.data, pretrained=args.model, device=args.device, imgsz=args.imgsz, batch=args.batch_size, max_det=1, plots=True)


if __name__ == '__main__':
   main()
