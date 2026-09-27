"""
页面2: 拍照识别 + 智能多轮问答
拍照/上传图像 → YOLO 检测 → 用户提问 → Qwen-VL 回答
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from backend.config import (
    YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD, get_dashscope_api_key,
    init_session_state
)
from backend.yolo_detector import YOLODetector
from backend.dashscope_client import DashScopeClient
from backend.utils import pil_to_cv2, cv2_to_pil
from backend.logging_setup import setup_logging


def get_yolo_detector():
    """获取或创建 YOLO 检测器"""
    if st.session_state.yolo_detector is None:
        try:
            st.session_state.yolo_detector = YOLODetector(
                model_path=YOLO_MODEL_PATH,
                conf_threshold=YOLO_CONF_THRESHOLD
            )
        except Exception as e:
            st.error(f"YOLO 模型加载失败: {e}")
            return None
    return st.session_state.yolo_detector


def get_dashscope_client():
    """获取或创建 DashScope 客户端"""
    if st.session_state.dashscope_client is None:
        st.session_state.dashscope_client = DashScopeClient()
    return st.session_state.dashscope_client


def detect_image(image: Image.Image):
    """对图像执行 YOLO 检测"""
    detector = get_yolo_detector()
    if detector is None:
        return None, []

    cv2_img = pil_to_cv2(image)
    detections = detector.detect(cv2_img)
    annotated = detector.draw_boxes(cv2_img, detections)
    annotated_pil = cv2_to_pil(annotated)

    return annotated_pil, detections


def main():
    setup_logging()
    init_session_state()
    st.title("拍照识别 + 智能问答")
    st.markdown("拍照或上传图像 → YOLO 目标检测 → 智能语义问答")

    # ==================== 布局：左侧图像，右侧问答 ====================
    col_img, col_qa = st.columns([1, 1])

    with col_img:
        st.subheader("图像输入")

        # 图像来源选择
        source = st.radio("图像来源", ["上传文件", "摄像头拍照"], horizontal=True)

        image = None
        if source == "上传文件":
            uploaded = st.file_uploader(
                "选择图片", type=["jpg", "jpeg", "png", "bmp"],
                help="支持 JPG、PNG、BMP 格式"
            )
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
        else:
            camera_image = st.camera_input("拍照")
            if camera_image:
                image = Image.open(camera_image).convert("RGB")

        # 显示原始图像
        if image:
            st.image(image, caption="原始图像", width="stretch")

            # 执行 YOLO 检测
            with st.spinner("YOLO 检测中..."):
                annotated_pil, detections = detect_image(image)
                st.session_state.qa_image = image
                st.session_state.qa_detections = detections

            # 显示检测结果
            if annotated_pil:
                st.image(annotated_pil, caption="YOLO 检测结果", width="stretch")

            # 显示检测摘要
            if detections:
                st.markdown("**检测到以下物体：**")
                from collections import Counter
                counts = Counter(d["class"] for d in detections)
                cols = st.columns(min(len(counts), 4))
                for i, (cls, cnt) in enumerate(counts.most_common()):
                    max_conf = max(d["confidence"] for d in detections if d["class"] == cls)
                    with cols[i % 4]:
                        st.metric(cls, f"{cnt} 个", f"max {max_conf:.0%}")

    with col_qa:
        st.subheader("智能问答")

        dashscope = get_dashscope_client()

        # 预设问题
        st.markdown("**快速提问**")
        question_templates = DashScopeClient.QUESTION_TEMPLATES
        template_tabs = st.tabs(list(question_templates.keys()))

        quick_question = None
        for tab, (category, questions) in zip(template_tabs, question_templates.items()):
            with tab:
                for q in questions:
                    if st.button(q, key=f"q_{category}_{q[:10]}", width="stretch"):
                        quick_question = q

        # 手动输入
        st.markdown("---")
        user_question = st.text_input(
            "输入你的问题", placeholder="例如：图里有几个人？",
            key="user_question_input"
        )

        # 处理提问
        question = user_question or quick_question
        if question and st.button("提问", type="primary", width="stretch"):
            image = st.session_state.qa_image
            detections = st.session_state.qa_detections

            if image is None:
                st.warning("请先上传或拍摄一张图片")
            elif not dashscope.is_initialized:
                st.warning("DashScope API Key 未配置，请先在侧边栏设置")
            else:
                # 生成检测摘要
                detector = get_yolo_detector()
                summary = detector.get_summary(detections) if detector else ""

                with st.spinner("Qwen-VL 思考中..."):
                    result = dashscope.analyze(
                        image_input=image,
                        prompt=question,
                        detection_summary=summary
                    )

                if result["success"]:
                    # 添加到聊天历史
                    st.session_state.chat_history.append({
                        "role": "user", "content": question
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": result["answer"]
                    })
                else:
                    st.error(f"回答失败: {result['error']}")

        # 显示聊天历史
        st.markdown("---")
        st.subheader("对话历史")

        if not st.session_state.chat_history:
            st.info("尚未开始对话，请先上传图片并提问")
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(msg["content"])

        # 清除历史
        if st.session_state.chat_history and st.button("清除对话历史", width="stretch"):
            st.session_state.chat_history = []
            st.rerun()


if __name__ == "__main__":
    main()
