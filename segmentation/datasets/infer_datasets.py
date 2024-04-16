from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import numpy as np

from segmentation_train_and_inference.utils.infer_utils import get_label_color_seg

class SegmentationInferenceDataset(Dataset):
    def __init__(self, csv_file, input_root_dir, csv_path_col_name, transform=None, label_col_name=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.input_root_dir = input_root_dir
        self.csv_path_col_name = csv_path_col_name
        self.csv_label_col_name = label_col_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.loc[idx, self.csv_path_col_name]
        
        label_seg = '' if self.csv_label_col_name is None else get_label_color_seg(self.data.loc[idx, self.csv_label_col_name])

        if self.input_root_dir is not None:
            image_orig = np.array(Image.open(os.path.join(self.input_root_dir, img_path)).convert('RGB'))
        else:
            image_orig = np.array(Image.open(img_path).convert('RGB'))

        image = None
        if self.transform:
            image = self.transform(image=image_orig)['image'].transpose(2,0,1)

        return image, img_path, label_seg, idx
