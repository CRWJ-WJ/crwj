import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import numpy as np
import torch
import torch.nn as nn
from config import BEST_MODEL_PATH, PROCESSED_DIR, IMG_SIZE
from model.detect_model import DefectCNN
from read_dataset import read_neu_dataset

model = DefectCNN(num_classes=6)
ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

all_data = read_neu_dataset()

# 收集每一类取一张
selected = dict()
for item in all_data:
    lab = item["label"]
    if lab not in selected:
        selected[lab] = item
    if len(selected)>=6:
        break

for lab,item in selected.items():
    img_path = os.path.join(PROCESSED_DIR, item["save_name"])
    label_true = lab

    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img,(IMG_SIZE, IMG_SIZE))
    tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)/255.0

    with torch.no_grad():
        out = model(tensor)
        pred = torch.argmax(torch.softmax(out,1)).item()
    print(f"真实标签:{label_true}, 模型预测:{pred}")
