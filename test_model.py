import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from model.detect_model import DefectCNN

# 不加载任何权重，完全随机初始化
model = DefectCNN(num_classes=6)
model.eval()

#造两张完全不一样的模拟灰度图片
img1 = torch.randn(1,1,200,200)
img2 = torch.randn(1,1,200,200)

with torch.no_grad():
    out1 = model(img1)
    p1 = torch.argmax(torch.softmax(out1,1)).item()
    out2 = model(img2)
    p2 = torch.argmax(torch.softmax(out2,1)).item()

print(f"随机图1预测:{p1}")
print(f"随机图2预测:{p2}")
