"""
树莓派智能视觉系统 - Streamlit 主入口
多页面应用入口，配置全局布局、侧边栏导航和全局状态初始化
"""

import sys
import os

# 将项目根目录加入 Python 路径，确保 backend 模块可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from backend.config import (
    STREAMLIT_PAGE_TITLE, STREAMLIT_PAGE_ICON, STREAMLIT_LAYOUT,
    get_dashscope_api_key, DASHSCOPE_MODEL,
    YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD,
    CAMERA_DEVICE_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    ALERT_CLASSES, ALERT_SOUND_PATH, SNAPSHOTS_DIR, DATABASE_PATH, BASE_DIR,
    EVENT_CONSECUTIVE_FRAMES, EVENT_MIN_INTERVAL, COCO_CLASSES,
    init_session_state
)
from backend.logging_setup import setup_logging

def main():
    """主入口"""
    # 日志初始化（须在其它模块打日志之前）
    setup_logging()

    # 页面配置
    st.set_page_config(
        page_title=STREAMLIT_PAGE_TITLE,
        page_icon=STREAMLIT_PAGE_ICON,
        layout=STREAMLIT_LAYOUT,
        initial_sidebar_state="expanded"
    )

    # 初始化 session_state
    init_session_state()

    # ==================== 侧边栏 ====================
    with st.sidebar:
        st.title("树莓派智能视觉系统")
        st.markdown("---")

        # API 配置
        st.subheader("DashScope API 配置")
        current_key = get_dashscope_api_key()
        api_key_input = st.text_input(
            "API Key",
            type="password",
            value=current_key if current_key != "your-api-key-here" else "",
            placeholder="输入 DashScope API Key",
            help="从阿里云百炼控制台获取。也可创建 .env 文件写入 DASHSCOPE_API_KEY=xxx"
        )
        if current_key == "your-api-key-here":
            st.caption(
                "未检测到 API Key。请在项目根目录创建 `.env` 文件，内容:\n"
                f"`DASHSCOPE_API_KEY=你的密钥`\n"
                f"文件位置: `{os.path.join(BASE_DIR, '.env')}`"
            )
        model_choice = st.selectbox(
            "模型选择",
            ["qwen-vl-plus", "qwen-vl-max"],
            index=0 if DASHSCOPE_MODEL == "qwen-vl-plus" else 1,
            help="qwen-vl-plus: 性价比高; qwen-vl-max: 最强能力"
        )

        # YOLO 配置
        st.markdown("---")
        st.subheader("YOLO 检测参数")
        conf_threshold = st.slider(
            "置信度阈值",
            min_value=0.1, max_value=0.9, value=YOLO_CONF_THRESHOLD, step=0.05,
            help="低于此置信度的检测结果将被过滤"
        )
        iou_threshold = st.slider(
            "NMS IoU 阈值",
            min_value=0.1, max_value=0.9, value=YOLO_IOU_THRESHOLD, step=0.05,
            help="非极大值抑制的交并比阈值"
        )

        # 摄像头配置
        st.markdown("---")
        st.subheader("摄像头设置")

        # 读取当前摄像头配置，计算默认索引
        cam_config = st.session_state.get("camera_config", {})
        current_device = cam_config.get("device_id", CAMERA_DEVICE_ID)
        current_picam = cam_config.get("use_picamera2", False)
        if current_picam:
            default_idx = 2
        elif current_device == 1:
            default_idx = 1
        else:
            default_idx = 0

        camera_options = ["USB 摄像头 (0)", "USB 摄像头 (1)", "CSI 摄像头 (picamera2)"]

        # 首次加载时用 camera_config 初始化 widget 状态，后续由 Streamlit 自动管理
        if "camera_selector" not in st.session_state:
            st.session_state.camera_selector = camera_options[default_idx]

        camera_source = st.selectbox(
            "视频源",
            camera_options,
            key="camera_selector"
        )
        if "CSI" in camera_source:
            camera_device_id = 0
            use_picamera2 = True
        else:
            camera_device_id = int(camera_source.split("(")[1][0])
            use_picamera2 = False

        # 告警设置
        st.markdown("---")
        st.subheader("告警设置")
        enable_alert = st.checkbox("启用告警声音", value=True)
        alert_classes = st.multiselect(
            "告警目标类别",
            options=COCO_CLASSES,
            default=[c for c in ALERT_CLASSES if c in COCO_CLASSES],
            help="检测到这些类别时播放告警声音"
        )

        # 保存配置
        st.markdown("---")
        if st.button("应用配置", width="stretch"):
            messages = []

            # --- 1) DashScope API Key ---
            if api_key_input and api_key_input != get_dashscope_api_key():
                os.environ["DASHSCOPE_API_KEY"] = api_key_input
                st.session_state.dashscope_client = None
                messages.append("API Key 已更新")

            # --- 2) DashScope 模型 ---
            current_model = st.session_state.get("dashscope_model", DASHSCOPE_MODEL)
            if model_choice != current_model:
                st.session_state.dashscope_model = model_choice
                st.session_state.dashscope_client = None  # 模型变了要重建客户端
                messages.append(f"模型已切换为 {model_choice}")

            # --- 3) YOLO 参数 ---
            if st.session_state.yolo_detector is not None:
                old_conf = st.session_state.yolo_detector.conf_threshold
                old_iou = st.session_state.yolo_detector.iou_threshold
                if conf_threshold != old_conf or iou_threshold != old_iou:
                    st.session_state.yolo_detector.conf_threshold = conf_threshold
                    st.session_state.yolo_detector.iou_threshold = iou_threshold
                    messages.append(f"YOLO 参数已更新 (conf={conf_threshold}, iou={iou_threshold})")

            # --- 4) 摄像头切换 ---
            new_cam_config = {
                "device_id": camera_device_id,
                "use_picamera2": use_picamera2,
            }
            old_cam_config = st.session_state.get("camera_config", {})
            cam_changed = (
                old_cam_config.get("device_id") != new_cam_config["device_id"]
                or old_cam_config.get("use_picamera2") != new_cam_config["use_picamera2"]
            )

            if cam_changed:
                # 先停止检测
                was_running = st.session_state.detection_running
                st.session_state.detection_running = False
                # 释放旧摄像头
                if st.session_state.camera:
                    st.session_state.camera.stop()
                    st.session_state.camera = None
                # 用新配置创建摄像头
                try:
                    from backend.camera_capture import CameraCapture
                    new_cam = CameraCapture(
                        device_id=new_cam_config["device_id"],
                        use_picamera2=new_cam_config["use_picamera2"]
                    )
                    new_cam.start()
                    st.session_state.camera = new_cam
                    st.session_state.camera_config = new_cam_config
                    messages.append(f"摄像头已切换为 {camera_source}")
                    if was_running:
                        st.session_state.detection_running = True
                except Exception as e:
                    st.session_state.camera_config = new_cam_config  # 仍保存配置
                    st.error(f"摄像头切换失败: {e}")
            else:
                st.session_state.camera_config = new_cam_config

            # --- 5) 告警设置 ---
            st.session_state.alert_config = {
                "enabled": enable_alert,
                "classes": list(alert_classes),  # 转 list 避免 multiselect 引用问题
            }
            if enable_alert:
                messages.append(f"告警已启用 (目标: {', '.join(alert_classes[:5])}" 
                                + f"{'...' if len(alert_classes) > 5 else ''})")
            else:
                messages.append("告警已关闭")

            # 汇总提示
            if messages:
                st.success(" | ".join(messages))
            else:
                st.info("配置无变化")

            st.rerun()

        # 系统信息
        st.markdown("---")
        st.caption(f"数据库: `{DATABASE_PATH}`")
        st.caption(f"快照目录: `{SNAPSHOTS_DIR}`")

    # ==================== 主页内容 ====================
    st.title("树莓派智能视觉系统")
    st.markdown(
        """
        基于 **YOLOv11 + DashScope Qwen-VL** 的智能视觉系统，运行于树莓派5上。

        **功能导航**：请从左侧边栏选择页面
        - **实时检测** — 摄像头实时视频流 + YOLO 目标检测
        - **拍照问答** — 拍照识别 + 智能多轮问答
        - **数据分析** — 日/周/月检测统计图表
        - **历史回放** — 事件时间线 + 快照查看
        """
    )

    # 快速状态面板
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("YOLO 模型", "yolov11n" if os.path.exists(YOLO_MODEL_PATH) else "待下载")
    with col2:
        dash_ready = bool(api_key_input and api_key_input != "your-api-key-here")
        st.metric("DashScope", "已就绪" if dash_ready else "未配置")
    with col3:
        st.metric("数据库状态", "已连接" if os.path.exists(DATABASE_PATH) else "待初始化")
    with col4:
        st.metric("摄像头", f"{camera_source}")


if __name__ == "__main__":
    main()
