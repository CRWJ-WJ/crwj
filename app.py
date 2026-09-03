# -*- coding: utf-8 -*-
"""
app.py
金属零部件表面缺陷智能检测与分级系统 —— Flask 后端主程序（B/S 架构）

技术方向覆盖：
  ① 计算机视觉：OpenCV 图像预处理 / 边缘检测 / ROI 提取（model/detect_model.py）
  ② 深度学习：PyTorch CNN 缺陷分类（model/detect_model.py）
  ③ 智能优化算法：遗传算法自适应优化视觉检测阈值（model/grade_decision.py）
数据库：SQLite（轻量、免安装，方案设计文档要求）

运行：python app.py  →  浏览器访问 http://127.0.0.1:5000
"""
import os
import sqlite3
import time
import traceback

from flask import Flask, jsonify, render_template, request

from model.detect_model import DefectDetectModel
from model.grade_decision import GAOptimizer, quality_grade_judge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 允许上传的图片格式
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# 数据库路径：可通过环境变量覆盖（自动化测试时指向临时库）
DB_PATH = os.environ.get(
    "METAL_INSPECT_DB",
    os.path.join(BASE_DIR, "database", "metal_inspect.db"),
)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 检测模型：存在 model/best.pt 时自动加载 CNN；否则降级为 OpenCV 视觉规则模式
detect_model = DefectDetectModel(weight_path=os.path.join(BASE_DIR, "model", "best.pt"))


# ================================ 数据库 ================================
def get_db_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化四张表：零件 / 检测记录 / 缺陷明细 / 算法参数。"""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS part_info (
            part_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            part_no     TEXT NOT NULL,
            part_name   TEXT DEFAULT '金属零部件',
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS inspect_record (
            record_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            part_id      INTEGER NOT NULL,
            quality_grade TEXT,
            total_area   INTEGER DEFAULT 0,
            defect_count INTEGER DEFAULT 0,
            img_path     TEXT,
            inspect_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (part_id) REFERENCES part_info(part_id)
        );

        CREATE TABLE IF NOT EXISTS defect_detail (
            detail_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id    INTEGER NOT NULL,
            defect_class TEXT,
            area_pixel   INTEGER,
            x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
            FOREIGN KEY (record_id) REFERENCES inspect_record(record_id)
        );

        CREATE TABLE IF NOT EXISTS algo_params (
            param_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            param_name  TEXT,
            param_value REAL,
            source      TEXT DEFAULT 'GA',
            update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()


init_db()


# ================================ 页面 ================================
@app.route("/")
def index():
    return render_template("index.html")


# ================================ 业务接口 ================================
@app.route("/api/upload_detect", methods=["POST"])
def api_upload_detect():
    """上传零件图 → 缺陷检测 → 质量分级 → 入库 → 返回结果。"""
    try:
        file = request.files.get("image_file")
        if file is None or file.filename == "":
            return jsonify({"code": -1, "msg": "未选择图片"})
        part_no = request.form.get("part_no", "UNKNOWN")

        # 安全保存上传文件（过滤路径穿越 + 校验类型 + 防止同名覆盖）
        fname = os.path.basename(file.filename)
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ALLOWED_EXTS:
            return jsonify({"code": -1, "msg": f"不支持的图片格式：{ext or '未知'}"})
        save_name = f"{int(time.time() * 1000)}_{fname}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], save_name)
        file.save(save_path)

        # ①缺陷检测（OpenCV 定位 + CNN 分类）
        defect_list, out_img_path = detect_model.predict_image(save_path)
        # ②质量分级
        grade, total_area, count = quality_grade_judge(defect_list)

        # ③数据库持久化（零件 / 记录 / 缺陷明细）
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO part_info(part_no, part_name) VALUES(?, ?)",
            (part_no, "金属零部件"),
        )
        part_id = cur.lastrowid
        cur.execute(
            "INSERT INTO inspect_record(part_id, quality_grade, total_area, defect_count, img_path) "
            "VALUES(?, ?, ?, ?, ?)",
            (
                part_id,
                grade,
                total_area,
                count,
                os.path.relpath(save_path, BASE_DIR).replace("\\", "/"),
            ),
        )
        record_id = cur.lastrowid
        for d in defect_list:
            cur.execute(
                "INSERT INTO defect_detail(record_id, defect_class, area_pixel, x1, y1, x2, y2) "
                "VALUES(?, ?, ?, ?, ?, ?, ?)",
                (
                    record_id,
                    d["defect_class"],
                    d["area_pixel"],
                    d["x1"],
                    d["y1"],
                    d["x2"],
                    d["y2"],
                ),
            )
        conn.commit()
        cur.close()
        conn.close()

        return jsonify(
            {
                "code": 0,
                "msg": "检测完成",
                "grade": grade,
                "defect_count": count,
                "total_area": total_area,
                "defect_list": defect_list,
                "out_img": out_img_path,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"code": -1, "msg": f"异常：{str(e)}"})


@app.route("/api/get_records", methods=["GET"])
def api_get_records():
    """查询历史质检记录。"""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT r.record_id, p.part_no, p.part_name, r.quality_grade,
               r.total_area, r.defect_count, r.img_path, r.inspect_time
        FROM inspect_record r
        JOIN part_info p ON r.part_id = p.part_id
        ORDER BY r.inspect_time DESC
        LIMIT 100
        """
    )
    records = [dict(row) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return jsonify({"records": records})


@app.route("/api/params", methods=["GET"])
def api_params():
    """查看当前视觉检测参数。"""
    return jsonify({"code": 0, "params": detect_model.thresholds})


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    """遗传算法自适应优化检测阈值，并将最优参数写回检测模块与数据库（闭环）。"""
    try:
        data = request.get_json(force=True, silent=True) or {}
        data_dir = os.path.join(BASE_DIR, "data", "NEU-DET")
        ga = GAOptimizer(
            images_dir=os.path.join(data_dir, "IMAGES"),
            ann_dir=os.path.join(data_dir, "ANNOTATIONS"),
            sample_size=int(data.get("sample_size", 15)),
        )
        best_params, history = ga.run(
            generations=int(data.get("generations", 20)),
            pop_size=int(data.get("pop_size", 15)),
        )

        # 闭环：写回检测模块阈值
        detect_model.set_thresholds(best_params)

        # 最优参数入库（可溯源）
        conn = get_db_conn()
        cur = conn.cursor()
        for k, v in best_params.items():
            cur.execute(
                "INSERT INTO algo_params(param_name, param_value, source) VALUES(?, ?, 'GA')",
                (k, float(v)),
            )
        conn.commit()
        cur.close()
        conn.close()

        return jsonify(
            {
                "code": 0,
                "msg": "遗传算法优化完成，参数已闭环写回",
                "best_params": best_params,
                "fitness": round(history[-1], 4) if history else 0.0,
                "history": [round(h, 4) for h in history],
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"code": -1, "msg": f"优化异常：{str(e)}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
