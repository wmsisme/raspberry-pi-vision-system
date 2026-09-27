"""
页面1: 实时视频流检测
摄像头实时画面 + YOLO 目标检测 + 告警 + 快照保存
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import threading

import cv2
import streamlit as st
import numpy as np

from backend.config import (
    YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD,
    CAMERA_DEVICE_ID, CAMERA_WIDTH, CAMERA_HEIGHT,
    SNAPSHOTS_DIR, ALERT_SOUND_PATH,
    EVENT_CONSECUTIVE_FRAMES, EVENT_MIN_INTERVAL, COCO_CLASSES,
    init_session_state
)
from backend.camera_capture import CameraCapture
from backend.yolo_detector import YOLODetector
from backend.database import DetectionDatabase
from backend.utils import save_snapshot, start_alert_sound, stop_alert_sound
from backend.logging_setup import setup_logging


def init_resources():
    """懒加载摄像头、YOLO、数据库。支持摄像头热切换"""
    # 摄像头: 如果不存在、或配置已变更，则创建/重建
    cam_config = st.session_state.get("camera_config", {})
    expected_device = cam_config.get("device_id", CAMERA_DEVICE_ID)
    expected_picam = cam_config.get("use_picamera2", False)

    need_new_camera = st.session_state.camera is None
    if not need_new_camera:
        # 检查当前摄像头是否与配置匹配（判断简单属性）
        try:
            if (st.session_state.camera.device_id != expected_device
                    or st.session_state.camera.use_picamera2 != expected_picam):
                need_new_camera = True
        except AttributeError:
            need_new_camera = True

    if need_new_camera:
        if st.session_state.camera:
            st.session_state.camera.stop()
            st.session_state.camera = None
        try:
            cam = CameraCapture(
                device_id=expected_device,
                use_picamera2=expected_picam
            )
            cam.start()
            st.session_state.camera = cam
            # 同步更新 session_state 中的摄像头配置，保持状态一致
            st.session_state.camera_config = {
                "device_id": expected_device,
                "use_picamera2": expected_picam,
            }
        except Exception as e:
            st.error(f"摄像头初始化失败: {e}")
            return False

    # YOLO 检测器
    if st.session_state.yolo_detector is None:
        try:
            detector = YOLODetector(
                model_path=YOLO_MODEL_PATH,
                conf_threshold=YOLO_CONF_THRESHOLD
            )
            st.session_state.yolo_detector = detector
        except Exception as e:
            st.error(f"YOLO 模型加载失败: {e}")
            return False

    # 数据库
    if st.session_state.database is None:
        try:
            st.session_state.database = DetectionDatabase()
        except Exception as e:
            st.warning(f"数据库初始化失败: {e}")

    return True


def release_resources():
    """释放资源"""
    if st.session_state.camera:
        st.session_state.camera.stop()
        st.session_state.camera = None
    st.session_state.detection_running = False


def detection_loop(placeholder, info_placeholder, save_auto: bool = False):
    """检测主循环（运行在 Streamlit 的循环中，非独立线程）

    参数:
        save_auto: 是否在告警触发时自动保存快照（对应控制栏的"自动保存快照"）
    """
    camera = st.session_state.camera
    detector = st.session_state.yolo_detector
    db = st.session_state.database

    # 事件去重：记录每个类别上次触发告警的时间
    last_event_time = {}
    consecutive_counter = {}  # 连续检测帧计数

    alert_config = st.session_state.get("alert_config", {
        "enabled": True,
        "classes": ["person", "dog", "cat"]
    })

    frame_count = 0
    start_time = time.time()
    fps = 0

    while st.session_state.detection_running:
        loop_start = time.time()

        # 获取最新帧
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        # YOLO 检测
        detections = detector.detect(frame)
        annotated = detector.draw_boxes(frame, detections)

        # 保存到 session_state 供其他页面使用
        st.session_state.current_frame = frame
        st.session_state.latest_detections = detections
        st.session_state.annotated_frame = annotated

        # 更新 FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        # 显示视频帧 (BGR → RGB)
        display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        placeholder.image(display_frame, channels="RGB", width="stretch")

        # 检测结果文本摘要
        detector_summary = detector.get_summary(detections)
        info_placeholder.text(
            f"FPS: {fps:.1f} | 检测到 {len(detections)} 个物体\n{detector_summary}"
        )

        # 保存检测结果到数据库
        if db and detections:
            db.save_detections(detections)

        # 告警检测
        if alert_config.get("enabled", True):
            alert_classes = set(alert_config.get("classes", []))
            now = time.time()

            # 收集当前帧中出现的高置信度告警类别
            detected_alert_classes = set()
            for det in detections:
                if det["class"] in alert_classes and det["confidence"] >= 0.5:
                    detected_alert_classes.add(det["class"])

            # 未在本帧出现的类别，重置连续计数（避免"断帧后残留"）
            for k in list(consecutive_counter.keys()):
                if k not in detected_alert_classes:
                    consecutive_counter[k] = 0

            # 对每个检测到的告警类更新连续帧计数
            for cls in detected_alert_classes:
                consecutive_counter[cls] = consecutive_counter.get(cls, 0) + 1

            # 判断是否有类别达到连续帧阈值
            triggered_classes = {
                cls for cls, cnt in consecutive_counter.items()
                if cnt >= EVENT_CONSECUTIVE_FRAMES
            }

            # -- 声音控制：有触发类就循环播放，全部消失就停止 --
            if triggered_classes:
                start_alert_sound(ALERT_SOUND_PATH)
            else:
                stop_alert_sound()

            # -- 快照 / 数据库事件（依然用最小间隔去重） --
            for cls in triggered_classes:
                last_time = last_event_time.get(cls, 0)
                if now - last_time < EVENT_MIN_INTERVAL:
                    continue

                last_event_time[cls] = now

                # 找到该类别中置信度最高的检测结果
                matching = [d for d in detections if d["class"] == cls]
                best_det = max(matching, key=lambda d: d["confidence"]) if matching else None
                if best_det is None:
                    continue

                # 保存快照（仅在勾选"自动保存快照"时；事件记录本身始终写入数据库）
                snapshot_path = None
                if save_auto:
                    snapshot_path = save_snapshot(annotated, SNAPSHOTS_DIR, f"alert_{cls}")

                # 保存事件到数据库
                if db:
                    db.save_event(
                        event_type="alert",
                        class_name=cls,
                        description=f"检测到 {cls}，置信度 {best_det['confidence']:.2f}",
                        confidence=best_det["confidence"],
                        snapshot_path=snapshot_path
                    )

        # 控制循环速率（目标 ~10 FPS）
        elapsed_loop = time.time() - loop_start
        if elapsed_loop < 0.08:
            time.sleep(0.08 - elapsed_loop)


def main():
    setup_logging()
    init_session_state()

    # ==================== 侧边栏：摄像头切换 ====================
    with st.sidebar:
        st.subheader("摄像头设置")

        # 读取当前摄像头配置
        cam_config = st.session_state.get("camera_config", {})
        current_device = cam_config.get("device_id", CAMERA_DEVICE_ID)
        current_picam = cam_config.get("use_picamera2", False)

        # 构造当前选中项的索引（仅用于首次初始化 widget 状态）
        if current_picam:
            default_index = 2  # CSI 摄像头
        elif current_device == 1:
            default_index = 1  # USB 摄像头 (1)
        else:
            default_index = 0  # USB 摄像头 (0)

        camera_options = ["USB 摄像头 (0)", "USB 摄像头 (1)", "CSI 摄像头 (picamera2)"]

        # 首次加载时用 camera_config 初始化 widget 状态，后续由 Streamlit 自动管理
        if "camera_selector" not in st.session_state:
            st.session_state.camera_selector = camera_options[default_index]

        selected_camera = st.selectbox(
            "视频源",
            camera_options,
            key="camera_selector",
            help="切换摄像头源，切换后需重新启动检测"
        )

        # 解析用户选择
        if "CSI" in selected_camera:
            new_device_id = 0
            new_use_picam = True
        else:
            new_device_id = int(selected_camera.split("(")[1][0])
            new_use_picam = False

        new_cam_config = {
            "device_id": new_device_id,
            "use_picamera2": new_use_picam,
        }

        # 检测摄像头选择是否发生变化
        cam_changed = (
            current_device != new_device_id
            or current_picam != new_use_picam
        )

        if cam_changed:
            # 停止当前检测
            was_running = st.session_state.detection_running
            st.session_state.detection_running = False
            # 释放旧摄像头
            if st.session_state.camera:
                st.session_state.camera.stop()
                st.session_state.camera = None
            # 更新配置
            st.session_state.camera_config = new_cam_config

        st.caption(f"当前视频源: {selected_camera}")

    st.title("实时视频流检测")
    st.markdown("摄像头实时画面 + YOLO 目标检测，支持告警和快照保存")

    # ==================== 控制栏 ====================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not st.session_state.detection_running:
            if st.button("启动检测", type="primary", width="stretch"):
                if init_resources():
                    st.session_state.detection_running = True
                    st.rerun()
        else:
            if st.button("停止检测", type="secondary", width="stretch"):
                st.session_state.detection_running = False
                st.rerun()

    with col2:
        save_auto = st.checkbox("自动保存快照", value=False,
                                help="检测到告警目标时自动保存快照")

    with col3:
        manual_snap = st.button("手动截图", width="stretch",
                                disabled=not st.session_state.detection_running)

    with col4:
        st.metric("检测状态",
                  "运行中" if st.session_state.detection_running else "已停止")

    # 手动截图
    if manual_snap and st.session_state.annotated_frame is not None:
        path = save_snapshot(st.session_state.annotated_frame, SNAPSHOTS_DIR, "manual")
        st.success(f"截图已保存: {path}")

    # ==================== 检测显示区 ====================
    video_placeholder = st.empty()
    info_placeholder = st.empty()

    # ==================== 检测循环 ====================
    if st.session_state.detection_running:
        detection_loop(video_placeholder, info_placeholder, save_auto=save_auto)
        # 当检测停止时显示
        if st.session_state.annotated_frame is not None:
            display = cv2.cvtColor(st.session_state.annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display, channels="RGB", width="stretch")
    else:
        # 未运行时显示占位
        if st.session_state.annotated_frame is not None:
            display = cv2.cvtColor(st.session_state.annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display, channels="RGB", width="stretch")
        else:
            video_placeholder.info("点击「启动检测」开始实时视频流检测")

    # ==================== 最近检测结果表格 ====================
    if st.session_state.latest_detections:
        st.markdown("---")
        st.subheader("最近检测结果")
        import pandas as pd
        df = pd.DataFrame(st.session_state.latest_detections)
        if not df.empty:
            df_display = df[["class", "confidence", "bbox"]].copy()
            df_display.columns = ["类别", "置信度", "边界框"]
            df_display["置信度"] = df_display["置信度"].apply(lambda x: f"{x:.2%}")
            st.dataframe(df_display, width="stretch")


if __name__ == "__main__":
    main()
