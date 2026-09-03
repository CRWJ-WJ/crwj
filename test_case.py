# -*- coding: utf-8 -*-
"""
test_case.py
项目自带自动化测试（课程设计要求：项目须通过自带的自动化测试）。

覆盖：
  1. 质量分级决策（A/B/C 及空列表边界）
  2. 缺陷检测模型（OpenCV 模式端到端检测 + 参数闭环）
  3. 遗传算法优化模块（参数搜索 + 适应度历史）
  4. Flask 接口（首页、上传检测、历史记录、参数查看）
  5. 数据库读写（通过接口隐式验证，使用临时库避免污染）

运行：python test_case.py
"""
import os
import sys
import tempfile
import unittest

# 测试使用独立的临时数据库，避免污染正式数据
TEST_DB = os.path.join(tempfile.mkdtemp(prefix="metal_test_"), "test_metal_inspect.db")
os.environ["METAL_INSPECT_DB"] = TEST_DB

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import app  # noqa: E402   （导入时会初始化临时数据库）
from model.detect_model import DefectDetectModel  # noqa: E402
from model.grade_decision import GAOptimizer, quality_grade_judge  # noqa: E402

NEU_IMG_DIR = os.path.join(BASE, "data", "NEU-DET", "IMAGES")
NEU_ANN_DIR = os.path.join(BASE, "data", "NEU-DET", "ANNOTATIONS")


def first_image():
    """取数据集中的第一张图片。"""
    if not os.path.isdir(NEU_IMG_DIR):
        return None
    for f in sorted(os.listdir(NEU_IMG_DIR)):
        if f.lower().endswith(".jpg"):
            return os.path.join(NEU_IMG_DIR, f)
    return None


class TestGradeDecision(unittest.TestCase):
    """质量分级逻辑测试。"""

    def test_grade_A(self):
        grade, area, count = quality_grade_judge([{"area_pixel": 500}, {"area_pixel": 300}])
        self.assertEqual(grade, "A")
        self.assertEqual(area, 800)
        self.assertEqual(count, 2)

    def test_grade_B(self):
        grade, _, _ = quality_grade_judge([{"area_pixel": 1500}, {"area_pixel": 1500}, {"area_pixel": 1500}])
        self.assertEqual(grade, "B")

    def test_grade_C_by_area(self):
        grade, _, _ = quality_grade_judge([{"area_pixel": 7000}])
        self.assertEqual(grade, "C")

    def test_grade_C_by_count(self):
        grade, _, _ = quality_grade_judge([{"area_pixel": 100}] * 8)
        self.assertEqual(grade, "C")

    def test_empty_list(self):
        grade, area, count = quality_grade_judge([])
        self.assertEqual(area, 0)
        self.assertEqual(count, 0)
        self.assertEqual(grade, "A")


class TestDetectModel(unittest.TestCase):
    """缺陷检测模型测试（OpenCV 模式，无需训练权重）。"""

    def test_predict_end_to_end(self):
        img = first_image()
        self.assertIsNotNone(img, "未找到 NEU-DET 测试图片")
        model = DefectDetectModel()
        defects, out_img = model.predict_image(img)
        self.assertIsInstance(defects, list)
        self.assertGreaterEqual(len(defects), 1, "应至少检测出一个缺陷区域")
        first = defects[0]
        for key in ("defect_class", "area_pixel", "x1", "y1", "x2", "y2"):
            self.assertIn(key, first)
        self.assertGreater(first["area_pixel"], 0)
        # 标注图应已生成且可访问
        self.assertIsNotNone(out_img)
        self.assertTrue(os.path.exists(os.path.join(BASE, out_img.replace("/", os.sep))))

    def test_set_thresholds_closed_loop(self):
        model = DefectDetectModel()
        model.set_thresholds({"canny_low": 40, "area_min": 500, "blur_k": 5})
        self.assertEqual(model.thresholds["canny_low"], 40)
        self.assertEqual(model.thresholds["area_min"], 500)
        self.assertEqual(model.thresholds["blur_k"], 5)


class TestGAOptimizer(unittest.TestCase):
    """遗传算法优化模块测试。"""

    def test_run_returns_valid_params(self):
        ga = GAOptimizer(images_dir=NEU_IMG_DIR, ann_dir=NEU_ANN_DIR, sample_size=4)
        best, history = ga.run(generations=2, pop_size=4)
        for key in ("canny_low", "canny_high", "area_min", "blur_k"):
            self.assertIn(key, best)
        self.assertEqual(len(history), 2)
        self.assertTrue(all(0.0 <= h <= 1.0 for h in history))


class TestFlaskAPI(unittest.TestCase):
    """Flask 接口集成测试（使用测试客户端 + 临时数据库）。"""

    @classmethod
    def setUpClass(cls):
        app.app.config["TESTING"] = True
        cls.client = app.app.test_client()

    def test_index_page(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)

    def test_upload_detect_and_query(self):
        img = first_image()
        self.assertIsNotNone(img)
        with open(img, "rb") as f:
            r = self.client.post(
                "/api/upload_detect",
                data={"part_no": "TEST-001", "image_file": (f, os.path.basename(img))},
                content_type="multipart/form-data",
            )
        j = r.get_json()
        self.assertEqual(j["code"], 0, f"检测接口异常：{j.get('msg')}")
        self.assertIn("grade", j)
        self.assertIn("defect_list", j)

        # 历史记录应包含刚才那条
        r2 = self.client.get("/api/get_records")
        records = r2.get_json()["records"]
        self.assertTrue(any(x["part_no"] == "TEST-001" for x in records))

    def test_params_endpoint(self):
        r = self.client.get("/api/params")
        j = r.get_json()
        self.assertEqual(j["code"], 0)
        self.assertIn("canny_low", j["params"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
