import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, request, jsonify, render_template
import cv2
import torch
import numpy as np
from config import BEST_MODEL_PATH, IMG_SIZE, CN_DEFECT_CLASSES
from model.detect_model import DefectCNN

app = Flask(__name__)
app.config['DEBUG'] = False

model = DefectCNN(num_classes=6)
ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"code":400,"msg":"未接收到文件"})

        file_bytes = file.read()
        img_np = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({"code":500,"msg":"图片解码失败"})

        img = cv2.resize(img,(IMG_SIZE, IMG_SIZE))
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)/255.0
        with torch.no_grad():
            out = model(tensor)
            pred_idx = int(torch.argmax(out, dim=1).item())
            conf = float(torch.softmax(out,dim=1).max().item())

        defect_name = CN_DEFECT_CLASSES[pred_idx]
        print(f"预测索引:{pred_idx}, 置信度:{conf:.3f}, 缺陷:{defect_name}")

        return jsonify({
            "code":200,
            "label_idx":pred_idx,
            "confidence":conf,
            "defect":defect_name
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"code":500,"msg":str(e)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
