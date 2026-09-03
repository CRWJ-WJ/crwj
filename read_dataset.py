import os
import csv
from config import CSV_PATH, PROCESSED_DIR


def read_neu_dataset():
    """读取预处理后的数据集列表"""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"数据集CSV不存在: {CSV_PATH}\n请先运行 python data_preprocess.py 生成数据"
        )

    data_list = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 补全图片完整路径
            row["image_path"] = os.path.join(PROCESSED_DIR, row["save_name"])
            data_list.append(row)
    return data_list


if __name__ == "__main__":
    data = read_neu_dataset()
    print(f"读取样本总数：{len(data)}")
    print("前3条样本：")
    for d in data[:3]:
        print(f"  {d['cls_name']} - {d['save_name']}")
