"""
首次运行体验验证：models/ 为空时能否自动下载并缓存 YOLO 权重
（临时脚本，验证后即删。在 GitHub 克隆的副本里运行）
"""
import os
import sys
import tempfile
import time

CLONE = sys.argv[1]
sys.path.insert(0, CLONE)
os.chdir(tempfile.mkdtemp(prefix="firstrun_"))  # 故意用无关 CWD，模拟用户从任意目录启动

print("=" * 68)
print("首次运行验证：models/ 为空 → 自动下载 → 回写本地缓存")
print("=" * 68)

from backend.config import YOLO_MODEL_PATH, YOLO_MODEL_SHORT
from backend.logging_setup import setup_logging
setup_logging()

print(f"  克隆目录      : {CLONE}")
print(f"  期望本地路径  : {YOLO_MODEL_PATH}")
print(f"  启动前存在    : {os.path.exists(YOLO_MODEL_PATH)}")
print(f"  当前 CWD      : {os.getcwd()}  (与项目无关)")
print("-" * 68)

assert not os.path.exists(YOLO_MODEL_PATH), "前置条件不满足：本地不该有权重"

t0 = time.time()
from backend.yolo_detector import YOLODetector
det = YOLODetector()
elapsed = time.time() - t0

print("-" * 68)
print(f"  加载结果      : {'成功' if det.is_loaded else '失败'}")
print(f"  耗时          : {elapsed:.1f} 秒")
print(f"  实际模型路径  : {det.model_path}")
print(f"  本地缓存已生成: {os.path.exists(YOLO_MODEL_PATH)}")
if os.path.exists(YOLO_MODEL_PATH):
    print(f"  缓存文件大小  : {os.path.getsize(YOLO_MODEL_PATH)/1e6:.2f} MB")
print("-" * 68)

import numpy as np
res = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
print(f"  空图推理      : 返回 {len(res)} 个目标（0 属预期）")

ok = det.is_loaded and os.path.exists(YOLO_MODEL_PATH)
print("=" * 68)
print("结论:", "首次运行自动下载 + 本地缓存 全部正常" if ok else "首次运行链路存在问题")
sys.exit(0 if ok else 1)
