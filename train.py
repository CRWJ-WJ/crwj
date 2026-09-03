import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    CSV_PATH, PROCESSED_DIR, MODEL_DIR, BEST_MODEL_PATH,
    IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE
)
from read_dataset import read_neu_dataset
from model.detect_model import DefectCNN


class DefectDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(PROCESSED_DIR, item["save_name"])
        
        # 兼容中文路径：二进制读取 + OpenCV 解码
        try:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
        except Exception:
            img = None
        
        if img is None:
            # 兜底：读取失败时返回空白图，避免训练中断
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        label = int(item["label"])
        return img_tensor, label


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 读取数据并划分训练集/测试集（分层抽样，保证类别比例一致）
    all_data = read_neu_dataset()
    train_data, test_data = train_test_split(
        all_data, test_size=0.2, random_state=42,
        stratify=[d["label"] for d in all_data]
    )
    print(f"训练集：{len(train_data)}  测试集：{len(test_data)}")

    train_loader = DataLoader(
        DefectDataset(train_data), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        DefectDataset(test_data), batch_size=BATCH_SIZE, shuffle=False
    )

    # 模型、损失函数、优化器
    model = DefectCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"使用设备：{device}")

    for epoch in range(EPOCHS):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} 训练")
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # ===== 测试阶段 =====
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        avg_loss = train_loss / len(train_loader)
        print(
            f"Epoch {epoch+1:2d} | 训练损失: {avg_loss:.4f} | "
            f"测试准确率: {acc*100:.2f}%"
        )

        # 保存最优模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  → 最优模型已更新，准确率: {best_acc*100:.2f}%")

    print(f"\n训练完成！最高测试准确率: {best_acc*100:.2f}%")
    print(f"模型保存位置: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
