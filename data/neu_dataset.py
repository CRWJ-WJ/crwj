import os
import sys

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
from cv_utils import cv_imread


class NEUDataset(Dataset):
    """NEU 表面缺陷图像分类数据集（配合 data_preprocess.py 生成的目录结构）。"""

    def __init__(self, root, csv_file, transform=None):
        self.root = root
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = os.path.join(self.root, row["cls_name"], row["save_name"])
        img = cv_imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label


if __name__ == "__main__":
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200, 200)),
        transforms.ToTensor()
    ])
    train_set = NEUDataset(root="./neu_processed/train", csv_file="./neu_processed/train_label.csv", transform=trans)
    print(f"数据集长度 {len(train_set)}")
    img, lab = train_set[0]
    print(f"图片shape {img.shape}, label {lab}")
