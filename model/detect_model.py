import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import cv2
from config import IMG_SIZE


class DefectCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 25 * 25, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def load_weights(self, weight_path):
        self.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
        self.eval()


def predict_defect(model, img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).contiguous() / 255.0
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return predicted.item(), confidence.item()
