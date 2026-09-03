import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*25*25,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,num_classes)
        )
    def forward(self,x):
        x=self.features(x)
        x=x.flatten(1)
        return self.classifier(x)

model = SimpleCNN()
model.eval()

img1 = torch.randn(1,1,200,200)
img2 = torch.randn(1,1,200,200)

with torch.no_grad():
    o1=model(img1)
    o2=model(img2)
    p1=torch.argmax(torch.softmax(o1,1)).item()
    p2=torch.argmax(torch.softmax(o2,1)).item()
    print(f"o1 raw:{o1}")
    print(f"o2 raw:{o2}")
    print(f"pred1={p1}, pred2={p2}")
