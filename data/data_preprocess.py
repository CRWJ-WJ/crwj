import os
import cv2
import random
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# ===================== 配置参数（这里路径指向你现在这个NEU‑DET文件夹） =====================
DATASET_ROOT = "./NEU-DET"
IMG_FOLDER = os.path.join(DATASET_ROOT, "IMAGES")
ANN_FOLDER = os.path.join(DATASET_ROOT, "ANNOTATIONS")

OUTPUT_ROOT = "./neu_processed"
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]
IMG_SIZE = (200, 200)
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 创建输出目录
for split in ["train", "val", "test"]:
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, cls), exist_ok=True)


def image_preprocess_vision(img: np.ndarray) -> np.ndarray:
    """OpenCV预处理：灰度、降噪、ROI截取"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), sigmaX=1.2)
    h, w = gray.shape
    roi = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    roi_resize = cv2.resize(roi, IMG_SIZE)
    return roi_resize


def data_augmentation(img: np.ndarray) -> list:
    """数据增强"""
    aug_imgs = [img.copy()]
    aug_imgs.append(cv2.flip(img, 1))
    aug_imgs.append(cv2.flip(img, 0))
    aug_imgs.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16)+noise, 0, 255).astype(np.uint8)
    aug_imgs.append(noisy_img)
    return aug_imgs


def parse_xml(xml_path):
    """解析xml，拿到缺陷类别"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    obj = root.find("object")
    cls_name = obj.find("name").text
    return cls_name


def process_dataset():
    all_samples = []
    img_list = [f for f in os.listdir(IMG_FOLDER) if f.endswith(".jpg")]

    for img_name in img_list:
        img_path = os.path.join(IMG_FOLDER, img_name)
        xml_name = img_name.replace(".jpg", ".xml")
        xml_path = os.path.join(ANN_FOLDER, xml_name)

        if not os.path.exists(xml_path):
            continue
        cls_name = parse_xml(xml_path)
        if cls_name not in CLASS_NAMES:
            continue
        label_idx = CLASS_NAMES.index(cls_name)

        raw_img = cv2.imread(img_path)
        if raw_img is None:
            continue

        roi_img = image_preprocess_vision(raw_img)
        aug_list = data_augmentation(roi_img)

        for idx, aug_img in enumerate(aug_list):
            save_name = f"{cls_name}_{img_name.split('.')[0]}_{idx}.jpg"
            all_samples.append({
                "cls_name": cls_name,
                "label": label_idx,
                "save_name": save_name,
                "image_array": aug_img
            })

    df = pd.DataFrame(all_samples)
    train_df, temp_df = train_test_split(df, train_size=TRAIN_RATIO, random_state=RANDOM_SEED, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, train_size=VAL_RATIO/(VAL_RATIO+TEST_RATIO),
                                       random_state=RANDOM_SEED, stratify=temp_df["label"])

    def save_split(split_df, split_name):
        for _, row in split_df.iterrows():
            out_path = os.path.join(OUTPUT_ROOT, split_name, row["cls_name"], row["save_name"])
            cv2.imwrite(out_path, row["image_array"])
        csv_path = os.path.join(OUTPUT_ROOT, f"{split_name}_label.csv")
        split_df[["cls_name", "label", "save_name"]].to_csv(csv_path, index=False)

    save_split(train_df, "train")
    save_split(val_df, "val")
    save_split(test_df, "test")

    print(f"数据集处理完成！输出目录：{OUTPUT_ROOT}")
    print(f"训练集:{len(train_df)} 样本，验证集:{len(val_df)} 样本，测试集:{len(test_df)} 样本")


if __name__ == "__main__":
    process_dataset()