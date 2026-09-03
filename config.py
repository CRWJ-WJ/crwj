import os

# ====================== 仅需修改这一行即可切换项目路径 ======================
ROOT_DIR = r"C:\Users\荣耀\Desktop\git\crwj"
# ==========================================================================

# 原始数据集目录
DATA_ROOT = os.path.join(ROOT_DIR, "NEU-DET")
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
ANNOTATION_DIR = os.path.join(DATA_ROOT, "annotations")

# 预处理输出目录
PROCESSED_DIR = os.path.join(ROOT_DIR, "neu_processed")
CSV_PATH = os.path.join(PROCESSED_DIR, "dataset.csv")

# 模型保存目录
MODEL_DIR = os.path.join(ROOT_DIR, "model")
# 改回来！！不要best_merged.pt
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")



# Web上传临时目录
UPLOAD_DIR = os.path.join(ROOT_DIR, "upload")

# 缺陷类别
DEFECT_CLASSES = ["crazing", "inclusion", "patches",
                  "pitted_surface", "rolled-in_scale", "scratches"]
CN_DEFECT_CLASSES = ["裂纹", "夹杂", "斑块", "麻点", "氧化皮", "划痕"]

# 训练参数
IMG_SIZE = 200
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005

