# -*- coding: utf-8 -*-
"""
cv_utils.py
OpenCV 在 Windows 下无法直接读写含中文（非 ASCII）路径的文件。
这里用 numpy 内存读写 + imdecode/imencode 绕开该问题，统一供各模块调用。
"""
import os

import cv2
import numpy as np


def cv_imread(path, flags=cv2.IMREAD_COLOR):
    """支持中文路径的 cv2.imread。"""
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def cv_imwrite(path, img):
    """支持中文路径的 cv2.imwrite。"""
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok
