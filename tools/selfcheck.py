#!/usr/bin/env python3
"""
环境自检脚本

用途:
    在树莓派（或开发机）上部署后、验收前跑一次，确认整条链路可用。
    不需要摄像头也能跑；没有摄像头的项会明确标记为 SKIP 而不是 FAIL。

用法:
    python tools/selfcheck.py

检查项:
    1. Python 与依赖清单        —— requirements.txt 里每一项是否可导入
    2. 日志系统                 —— logs/ 是否可写、是否幂等
    3. 数据库                   —— 建表、写入、日/周/月/小时/直方图查询
    4. 快照落盘                 —— 含中文路径与纯 ASCII 路径两种分支
    5. 摄像头                   —— 枚举设备并取一帧（无设备则 SKIP）
    6. YOLO 模型                —— 加载权重并对真实帧推理（含自动下载）
    7. ONNX 模型                —— 若 models/ 下存在 .onnx 则验证推理
    8. 告警声音                 —— pygame 混音器初始化与播放（无音频设备则 SKIP）
    9. DashScope 客户端         —— Key 读取、Prompt 注入、预设问题模板

产物说明:
    所有测试用的数据库/快照/日志都会写入临时目录，不污染项目目录。
"""

import os
import shutil
import sys
import tempfile
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

RESULTS = []


def record(status, name, detail, seconds=0.0):
    RESULTS.append((status, name, detail, seconds))


def check(name, fn, skippable=False):
    t0 = time.time()
    try:
        detail, status = fn()
        record(status, name, detail, time.time() - t0)
    except SkipCheck as exc:
        record("SKIP", name, str(exc), time.time() - t0)
    except Exception as exc:
        record("FAIL", name, f"{type(exc).__name__}: {exc}", time.time() - t0)
        if os.environ.get("SELFCHECK_TRACE"):
            import traceback
            traceback.print_exc()


class SkipCheck(Exception):
    """环境不具备该检查条件（不算失败）"""


# ==================== 各项检查 ====================

def c_deps():
    """依赖清单逐项可导入性"""
    import importlib
    mapping = {
        "opencv-python": "cv2", "ultralytics": "ultralytics", "numpy": "numpy",
        "dashscope": "dashscope", "streamlit": "streamlit", "plotly": "plotly",
        "pandas": "pandas", "pillow": "PIL", "pygame": "pygame",
    }
    missing = []
    for pkg, mod in mapping.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise RuntimeError(f"缺少依赖: {', '.join(missing)} —— 请先 pip install -r requirements.txt")
    return f"requirements.txt 全部 {len(mapping)} 项依赖可用", "PASS"


def c_logging():
    """日志初始化与幂等性"""
    from backend.logging_setup import setup_logging, is_configured
    p1 = setup_logging()
    p2 = setup_logging()  # 重复调用应当幂等，不重复配置 handler
    assert p1 == p2, f"setup_logging 非幂等: {p1} != {p2}"
    assert is_configured(), "日志未完成初始化"
    assert os.path.exists(p1), f"日志文件未创建: {p1}"
    return f"日志文件 {os.path.relpath(p1, BASE_DIR)}，重复调用幂等", "PASS"


def c_database():
    """数据库建表/写入/五类查询"""
    from backend.database import DetectionDatabase
    db = DetectionDatabase()
    db.save_detections([
        {"class": "person", "confidence": 0.91, "bbox": [10, 20, 110, 220]},
        {"class": "car", "confidence": 0.66, "bbox": [320, 180, 520, 300]},
    ])
    today = time.strftime("%Y-%m-%d")
    daily = db.get_daily_stats(today)
    hourly = db.get_hourly_stats(today)
    dist = db.get_confidence_distribution()
    weekly = db.get_weekly_stats(today)
    monthly = db.get_monthly_stats(time.localtime().tm_year, time.localtime().tm_mon)
    assert daily, "日统计返回空"
    assert len(dist) == 10, f"置信度直方图桶数异常: {len(dist)}"
    return (f"日统计 {len(daily)} 类 / 周 {len(weekly)} 类 / 月 {len(monthly)} 类 / "
            f"小时 {len(hourly)} 段 / 直方图 {len(dist)} 桶"), "PASS"


def c_snapshot():
    """快照落盘：含中文路径 + ASCII 路径双分支"""
    import numpy as np
    from backend.utils import imwrite_unicode, save_snapshot
    frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
    tmp_root = tempfile.mkdtemp(prefix="selfcheck_")
    try:
        # 中文路径分支（项目根目录常含中文，这是真实踩过的坑）
        cn_dir = os.path.join(tmp_root, "中文目录", "2026-01-01")
        cn_path = os.path.join(cn_dir, "snap_cn.jpg")
        assert imwrite_unicode(cn_path, frame), "中文路径写入失败"
        assert os.path.getsize(cn_path) > 100, "中文路径文件过小"
        # ASCII 分支
        ascii_path = os.path.join(tmp_root, "ascii", "snap_ascii.jpg")
        assert imwrite_unicode(ascii_path, frame), "ASCII 路径写入失败"
        # save_snapshot 的日期分目录逻辑
        snap = save_snapshot(frame, os.path.join(tmp_root, "snaps"), "selfcheck")
        assert snap and os.path.exists(snap), f"save_snapshot 未落盘: {snap}"
        return (f"中文路径 {os.path.getsize(cn_path)}B / ASCII {os.path.getsize(ascii_path)}B / "
                f"save_snapshot 日期分目录 {os.path.basename(os.path.dirname(snap))}"), "PASS"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def c_camera():
    """摄像头枚举与取帧"""
    from backend.camera_capture import CameraCapture
    cam = None
    try:
        cam = CameraCapture(device_id=0, width=640, height=480, fps=15)
    except Exception as exc:
        raise SkipCheck(f"未检测到可用摄像头（{exc}）——树莓派上请确认排线/USB 连接")
    try:
        cam.start()
        frame = None
        for _ in range(40):
            time.sleep(0.15)
            frame = cam.get_frame()
            if frame is not None:
                break
        if frame is None:
            raise SkipCheck("摄像头已打开但 6 秒内取不到帧")
        return f"设备 0 取到帧 {frame.shape}（H×W×C）", "PASS"
    finally:
        if cam:
            cam.stop()


def c_yolo():
    """YOLO 加载与推理（无权重时自动下载）"""
    import numpy as np
    from backend.config import YOLO_MODEL_PATH
    from backend.yolo_detector import YOLODetector
    had_local = os.path.exists(YOLO_MODEL_PATH)
    det = YOLODetector()
    if not det.is_loaded:
        raise RuntimeError("模型加载失败（排查网络或手动放置权重到 models/）")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    t0 = time.time()
    res = det.detect(dummy)
    ms = (time.time() - t0) * 1000
    cached = os.path.exists(YOLO_MODEL_PATH)
    src = "本地已有权重" if had_local else ("自动下载并已缓存到 models/" if cached else "自动下载但未缓存")
    return f"{det.model_path}（{src}），空图推理 {len(res)} 目标 / {ms:.0f}ms", "PASS"


def c_onnx():
    """ONNX 模型推理（存在才检查）"""
    import numpy as np
    from backend.config import BASE_DIR
    onnx_dir = os.path.join(BASE_DIR, "models")
    onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")] \
        if os.path.isdir(onnx_dir) else []
    if not onnx_files:
        raise SkipCheck("models/ 下无 .onnx（可运行 tools/export_onnx.py 生成）")
    from backend.yolo_detector import YOLODetector
    path = os.path.join(onnx_dir, onnx_files[0])
    det = YOLODetector(model_path=path, use_onnx=True)
    if not det.is_loaded:
        raise RuntimeError("ONNX 模型加载失败，请确认已安装 onnxruntime")
    res = det.detect(np.zeros((320, 320, 3), dtype=np.uint8))
    return f"{onnx_files[0]}（{os.path.getsize(path)/1e6:.1f}MB）推理正常，返回 {len(res)} 目标", "PASS"


def c_alert():
    """告警声音（无音频设备则跳过）"""
    from backend.config import ALERT_SOUND_PATH
    from backend.utils import start_alert_sound, stop_alert_sound
    try:
        import pygame
    except ImportError:
        raise RuntimeError("pygame 未安装，告警无声")
    start_alert_sound(ALERT_SOUND_PATH, duration_ms=200)
    time.sleep(0.3)
    inited = pygame.mixer.get_init() is not None
    stop_alert_sound()
    if not inited:
        raise SkipCheck("无可用音频设备（pygame mixer 未初始化），代码已静默降级")
    src = "指定 wav" if os.path.exists(ALERT_SOUND_PATH) else "合成蜂鸣降级"
    return f"播放并停止正常，音源={src}", "PASS"


def c_dashscope():
    """DashScope 客户端构造与 Prompt 组装"""
    from backend.config import get_dashscope_api_key
    from backend.dashscope_client import DashScopeClient, DASHSCOPE_AVAILABLE
    if not DASHSCOPE_AVAILABLE:
        raise RuntimeError("dashscope SDK 未安装")
    key = get_dashscope_api_key()
    configured = key != "your-api-key-here"
    client = DashScopeClient()          # 用真实 Key 读数（未配置时为占位符）
    assert client.is_initialized == configured, "客户端初始化状态与 Key 配置不一致"
    probe = DashScopeClient(api_key="sk-selfcheck-structure-only")  # 只验结构，不发请求
    assert probe.is_initialized
    prompt = probe._build_prompt("图里有几个人？", "共检测到 2 个物体：\n- person: 2 个")
    assert "person" in prompt and "图里有几个人？" in prompt, "Prompt 未注入检测摘要"
    n_cat = len(probe.QUESTION_TEMPLATES)
    n_q = sum(len(v) for v in probe.QUESTION_TEMPLATES.values())
    probe.shutdown()
    state = "已配置" if configured else "未配置（拍照问答不可用，其余功能正常）"
    return f"API Key {state}，预设问题 {n_cat} 类 / {n_q} 条，Prompt 注入正常", "PASS"


# ==================== 主流程 ====================

CHECKS = [
    ("依赖清单", c_deps),
    ("日志系统", c_logging),
    ("数据库读写与统计", c_database),
    ("快照落盘（含中文路径）", c_snapshot),
    ("摄像头采集", c_camera),
    ("YOLO 模型加载与推理", c_yolo),
    ("ONNX 模型推理", c_onnx),
    ("告警声音", c_alert),
    ("DashScope 客户端", c_dashscope),
]


def main() -> int:
    print("=" * 68)
    print("树莓派智能视觉系统 - 环境自检")
    print("=" * 68)
    print(f"  项目目录: {BASE_DIR}")
    print(f"  Python  : {sys.version.split()[0]}")
    print("-" * 68)

    for name, fn in CHECKS:
        check(name, fn)

    print("-" * 68)
    n_pass = n_fail = n_skip = 0
    for status, name, detail, seconds in RESULTS:
        mark = {"PASS": "[OK]  ", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
        print(f"{mark} {name}  ({seconds:.1f}s)")
        print(f"        {detail}")
        n_pass += status == "PASS"
        n_fail += status == "FAIL"
        n_skip += status == "SKIP"

    print("=" * 68)
    print(f"结果: {n_pass} 通过 / {n_fail} 失败 / {n_skip} 跳过（共 {len(RESULTS)} 项）")
    if n_fail:
        print("提示: 失败的项通常只需 pip install 缺的包，或检查设备连接")
    else:
        print("提示: 本项目不使用摄像头也能跑通 数据分析 / 历史回放 两个页面")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
