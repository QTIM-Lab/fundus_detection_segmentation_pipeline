import argparse
from ultralytics import YOLO


def parse_args():
   parser = argparse.ArgumentParser(description='Train a YOLO model.')
   parser.add_argument('--model', type=str, required=True, help='Path to pretrained model file.')
   parser.add_argument('--data', type=str, required=True, help='Path to data config file (.yaml).')
   parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
   parser.add_argument('--imgsz', type=int, default=640, help='Input image size.')
   parser.add_argument('--lr0', type=float, default=0.000005, help='Initial learning rate.')
   parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
   parser.add_argument('--patience', type=int, default=7, help='Early stopping patience.')
   parser.add_argument('--train_resume', action='store_true', help='Resume training.')
   parser.add_argument('--device', type=int, default=1, help='Device.')
   parser.add_argument('--num_workers', type=int, default=8, help='Num Workers.')
   return parser.parse_args()


def main():
   args = parse_args()

   model = YOLO(args.model)  # Load pretrained model
   model.train(data=args.data, 
               pretrained=args.model,
               epochs=args.epochs, 
               imgsz=args.imgsz, 
               lr0=args.lr0, 
               batch=args.batch_size,
               patience=args.patience,
               device=args.device,
               verbose=True,
               optimizer="AdamW",
               resume=args.train_resume,
               workers=args.num_workers)


if __name__ == '__main__':
   main()
