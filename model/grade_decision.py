# -*- coding: utf-8 -*-
"""
model/grade_decision.py
质量分级决策 + 智能优化算法（技术方向③遗传算法）

职责：
1. quality_grade_judge()：根据缺陷列表（总面积、缺陷数）计算零件质量等级 A/B/C；
2. GAOptimizer：遗传算法自适应搜索最优视觉检测阈值
   （canny_low / canny_high / area_min / blur_k），
   以标注样本（NEU-DET 图像 + XML 框）上的“命中率”作为适应度，
   使检测参数适配不同批次工件表面条件，实现参数闭环优化。
"""
import os
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in os.sys.path:
    os.sys.path.insert(0, PROJECT_ROOT)
from cv_utils import cv_imread

# 默认分级边界（面积阈值 / 缺陷数阈值），可传入自定义边界
DEFAULT_GRADE_BOUNDARY = {
    "A_area": 2000, "A_count": 2,   # 总面积 <2000 且缺陷数 <=2 → A（优）
    "B_area": 6000, "B_count": 5,   # 总面积 <6000 且缺陷数 <=5 → B（良）
}


def quality_grade_judge(defect_list, boundary=None):
    """根据缺陷列表计算质量等级。

    Args:
        defect_list: [{"defect_class","area_pixel","x1","y1","x2","y2"}, ...]
        boundary: 分级边界字典（可选）

    Returns:
        (grade, total_area, count)
        grade: "A"（优）/ "B"（良）/ "C"（不合格）
    """
    b = boundary or DEFAULT_GRADE_BOUNDARY
    total_area = sum(int(d.get("area_pixel", 0)) for d in (defect_list or []))
    count = len(defect_list or [])

    if total_area < b["A_area"] and count <= b["A_count"]:
        grade = "A"
    elif total_area < b["B_area"] and count <= b["B_count"]:
        grade = "B"
    else:
        grade = "C"
    return grade, total_area, count


def _iou(box_a, box_b):
    """两个矩形框的 IoU。"""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (xa2 - xa1) * (ya2 - ya1))
    area_b = max(1, (xb2 - xb1) * (yb2 - yb1))
    return inter / (area_a + area_b - inter + 1e-6)


class GAOptimizer:
    """遗传算法：搜索最优视觉检测阈值。

    个体 = {canny_low, canny_high, area_min, blur_k}
    适应度 = 在标注样本上检测框与真实框(>IoU 0.3)的命中率 - 误检惩罚，取值 [0,1]。
    """

    # 参数搜索空间
    SEARCH_SPACE = {
        "canny_low": (30, 120, int),
        "canny_high_delta": (60, 180, int),  # high = low + delta
        "area_min": (50, 800, int),
        "blur_k": (3, 5, int),
    }

    def __init__(self, images_dir, ann_dir, sample_size=15, seed=42, iou_thresh=0.3):
        self.images_dir = images_dir
        self.ann_dir = ann_dir
        self.iou_thresh = iou_thresh
        random.seed(seed)
        self._prepare_samples(sample_size)

    # ---------------------------------------------------------------- 样本准备
    def _prepare_samples(self, sample_size):
        """选取 sample_size 张带标注图像作为适应度评估样本。"""
        xml_files = [f for f in os.listdir(self.ann_dir) if f.lower().endswith(".xml")]
        random.shuffle(xml_files)
        xml_files = xml_files[:sample_size]
        self.samples = []
        for xml_name in xml_files:
            img_name = xml_name.replace(".xml", ".jpg")
            img_path = os.path.join(self.images_dir, img_name)
            if not os.path.exists(img_path):
                continue
            boxes = self._parse_xml_boxes(os.path.join(self.ann_dir, xml_name))
            if not boxes:
                continue
            self.samples.append({"img_path": img_path, "boxes": boxes})
        if not self.samples:
            raise ValueError("GA 优化器：未找到可用的标注样本（检查数据目录）")

    @staticmethod
    def _parse_xml_boxes(xml_path):
        """解析 NEU-DET XML 标注，返回真实缺陷框列表 [(x1,y1,x2,y2), ...]。"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall("object"):
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            x1 = int(float(bnd.find("xmin").text))
            y1 = int(float(bnd.find("ymin").text))
            x2 = int(float(bnd.find("xmax").text))
            y2 = int(float(bnd.find("ymax").text))
            boxes.append((x1, y1, x2, y2))
        return boxes

    # ---------------------------------------------------------------- 检测与适应度
    def _detect(self, img, params):
        """用给定参数对图像做 Canny + 轮廓检测，返回检测框列表。"""
        low = int(params["canny_low"])
        high = int(params["canny_low"] + params["canny_high_delta"])
        area_min = float(params["area_min"])
        k = int(params["blur_k"]) | 1
        if k < 3:
            k = 3

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (k, k), 1.2)
        edges = cv2.Canny(blur, low, high)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < area_min:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
        return boxes

    def fitness(self, params):
        """适应度：真实框命中率 - 误检惩罚，取值 [0,1]。"""
        total_gt = 0
        total_hit = 0
        total_fp = 0
        for s in self.samples:
            img = cv_imread(s["img_path"])
            if img is None:
                continue
            det_boxes = self._detect(img, params)
            gt_boxes = s["boxes"]
            total_gt += len(gt_boxes)
            matched_gt = set()
            for db in det_boxes:
                best_iou = 0.0
                best_idx = -1
                for gi, gb in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    iou = _iou(db, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gi
                if best_idx >= 0 and best_iou >= self.iou_thresh:
                    matched_gt.add(best_idx)
            total_hit += len(matched_gt)
            total_fp += max(0, len(det_boxes) - len(matched_gt))

        if total_gt == 0:
            return 0.0
        recall = total_hit / total_gt
        fp_penalty = 0.1 * min(1.0, total_fp / max(1, total_gt))
        return max(0.0, min(1.0, recall - fp_penalty))

    # ---------------------------------------------------------------- 遗传算法
    def _random_individual(self):
        sp = self.SEARCH_SPACE
        return {
            "canny_low": random.randint(*sp["canny_low"][:2]),
            "canny_high_delta": random.randint(*sp["canny_high_delta"][:2]),
            "area_min": random.randint(*sp["area_min"][:2]),
            "blur_k": random.choice([3, 5]),
        }

    def _crossover(self, p1, p2):
        child = {}
        for key in p1:
            child[key] = p1[key] if random.random() < 0.5 else p2[key]
        return child

    def _mutate(self, ind, rate=0.15):
        sp = self.SEARCH_SPACE
        new = dict(ind)
        for key in ind:
            if random.random() < rate:
                lo, hi, _ = sp[key]
                new[key] = random.randint(lo, hi) if key != "blur_k" else random.choice([3, 5])
        return new

    def run(self, generations=20, pop_size=15, mutation_rate=0.15):
        """运行遗传算法，返回 (best_params, fitness_history)。"""
        population = [self._random_individual() for _ in range(pop_size)]
        fitness_history = []
        best_individual = None
        best_fitness = -1.0

        for gen in range(generations):
            scored = [(self.fitness(ind), ind) for ind in population]
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > best_fitness:
                best_fitness = scored[0][0]
                best_individual = dict(scored[0][1])
            fitness_history.append(scored[0][0])

            # 精英保留 + 锦标赛选择
            elites = [dict(ind) for _, ind in scored[: max(2, pop_size // 5)]]
            next_pop = list(elites)
            while len(next_pop) < pop_size:
                def pick():
                    a = random.choice(scored[: max(3, pop_size // 2)])[1]
                    b = random.choice(scored[: max(3, pop_size // 2)])[1]
                    return a, b
                p1, p2 = pick()
                child = self._mutate(self._crossover(p1, p2), mutation_rate)
                next_pop.append(child)
            population = next_pop

        # 转为检测模块可直接使用的阈值（去掉中间量 canny_high_delta）
        best_params = {
            "canny_low": int(best_individual["canny_low"]),
            "canny_high": int(best_individual["canny_low"] + best_individual["canny_high_delta"]),
            "area_min": int(best_individual["area_min"]),
            "blur_k": int(best_individual["blur_k"]),
        }
        return best_params, fitness_history


if __name__ == "__main__":
    data_dir = os.path.join(PROJECT_ROOT, "data", "NEU-DET")
    ga = GAOptimizer(
        images_dir=os.path.join(data_dir, "IMAGES"),
        ann_dir=os.path.join(data_dir, "ANNOTATIONS"),
        sample_size=10,
    )
    best, history = ga.run(generations=10, pop_size=10)
    print("最优参数:", best)
    print("适应度曲线:", [round(h, 4) for h in history])

    # 分级自检
    print("分级示例:", quality_grade_judge([{"area_pixel": 500}, {"area_pixel": 300}]))
