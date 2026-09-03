# -*- coding: utf-8 -*-
"""
database/init_db.py
独立数据库初始化脚本：创建 SQLite 数据库及四张表。
应用启动时也会自动调用 init_db()，本脚本供手动重建数据库使用。

用法：python database/init_db.py
"""
import os
import sqlite3
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "database", "metal_inspect.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS part_info (
    part_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    part_no     TEXT NOT NULL,
    part_name   TEXT DEFAULT '金属零部件',
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS inspect_record (
    record_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    part_id       INTEGER NOT NULL,
    quality_grade TEXT,
    total_area    INTEGER DEFAULT 0,
    defect_count  INTEGER DEFAULT 0,
    img_path      TEXT,
    inspect_time  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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


def init_db(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(SCHEMA)
    conn.commit()
    cur.close()
    conn.close()
    print(f"数据库初始化完成：{db_path}")


if __name__ == "__main__":
    init_db()
