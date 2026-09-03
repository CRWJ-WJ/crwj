import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import csv
import numpy as np
from config import (
    IMAGE_DIR, ANNOTATION_DIR, PROCESSED_DIR, CSV_PATH,
    DEFECT_CLASSES, IMG_SIZE
)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    csv_rows = []
    img_index = 0

    for label_idx, class_name in enumerate(DEFECT_CLASSES):
        cls_img_dir = os.path.join(IMAGE_DIR, class_name)
        if not os.path.isdir(cls_img_dir):
            print(f"警告：找不到 {cls_img_dir}，跳过")
            continue

        print(f"处理类别: {class_name}")

        # 支持 bmp、jpg、png 三种格式
        img_files = [
            f for f in os.listdir(cls_img_dir)
            if f.lower().endswith((".bmp", ".jpg", ".png"))
        ]

        for img_name in img_files:
            img_path = os.path.join(cls_img_dir, img_name)
            
            # 兼容中文路径：先读二进制字节，再解码图片
            try:
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
                img_np = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
            except Exception:
                continue

            if img is None:
                continue

            # 统一缩放尺寸
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # 保存处理后的图片（兼容中文路径）
            save_name = f"{class_name}_{img_index}.jpg"
            save_path = os.path.join(PROCESSED_DIR, save_name)
            try:
                ret, img_encoded = cv2.imencode('.jpg', img)
                if ret:
                    with open(save_path, 'wb') as f:
                        f.write(img_encoded.tobytes())
            except Exception:
                continue

            csv_rows.append({
                "save_name": save_name,
                "label": label_idx,
                "cls_name": class_name,
                "bboxes": ""
            })
            img_index += 1

    # 写入 CSV 标注文件
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["save_name", "label", "cls_name", "bboxes"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n预处理完成，共生成 {len(csv_rows)} 条样本")
    print(f"CSV 已保存至: {CSV_PATH}")


if __name__ == "__main__":
    main()
