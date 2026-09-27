"""
DashScope 云 API 客户端模块
封装 Qwen-VL 多模态模型调用，支持同步和异步（线程+队列）方式
"""

import base64
import io
import logging
import os
import threading
import time
from queue import Queue, Empty
from typing import Callable, Optional

from PIL import Image

from backend.config import (
    DASHSCOPE_MODEL, get_dashscope_api_key,
    API_TIMEOUT, API_MAX_RETRIES, API_IMAGE_MAX_WIDTH
)
from backend.utils import frame_to_base64, compress_image_for_api

logger = logging.getLogger(__name__)

# 阿里云 DashScope SDK
try:
    import dashscope
    from dashscope import MultiModalConversation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    logger.warning("dashscope 未安装，云端语义分析功能不可用")


class DashScopeClient:
    """
    DashScope 多模态对话客户端
    用于调用 Qwen-VL 模型进行图像理解和智能问答
    """

    # 预设问题模板
    QUESTION_TEMPLATES = {
        "数量统计": [
            "图里有几个人？",
            "图片中有几辆车？",
            "总共识别到了多少个物体？",
            "每个类别的数量分别是多少？",
        ],
        "存在性判断": [
            "图片里有猫吗？",
            "有没有椅子？",
            "有没有检测到手机？",
            "这个图片里有没有狗或者人？",
        ],
        "类别枚举": [
            "图中都有哪些类别的物体？",
            "识别到了哪些类型的东西？",
            "有什么目标被检测到了？",
        ],
        "位置关系": [
            "图片左边有些什么？",
            "中间识别到什么了？",
            "右下角有没有物体？",
            "哪些类别出现在画面顶部？",
        ],
        "面积占比": [
            "哪类物体面积最大？",
            "哪种物体占据画面最大的空间？",
            "哪个目标看起来最大？",
        ],
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        if api_key is None:
            api_key = get_dashscope_api_key()
        if model is None:
            # 尝试从 session_state 读取用户选择的模型
            try:
                import streamlit as st
                model = st.session_state.get("dashscope_model", DASHSCOPE_MODEL)
            except Exception:
                model = DASHSCOPE_MODEL
        self.api_key = api_key
        self.model = model
        self._initialized = False

        if DASHSCOPE_AVAILABLE and api_key and api_key != "your-api-key-here":
            dashscope.api_key = api_key
            self._initialized = True
            logger.info(f"DashScope 客户端初始化成功，模型: {model}")
        else:
            if not DASHSCOPE_AVAILABLE:
                logger.warning("dashscope SDK 未安装")
            elif not api_key or api_key == "your-api-key-here":
                logger.warning(
                    "DashScope API Key 未配置。请: 1) export DASHSCOPE_API_KEY=xxx "
                    "2) 或创建 .env 文件写入 DASHSCOPE_API_KEY=xxx  3) 或在 Streamlit 侧边栏输入"
                )

        # 异步调用队列
        self._task_queue: Queue = Queue(maxsize=10)
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_running = False

    # ==================== 同步调用（阻塞） ====================

    def analyze(
        self,
        image_input,
        prompt: str,
        detection_summary: str = "",
        retries: int = API_MAX_RETRIES
    ) -> dict:
        """
        同步调用 Qwen-VL 进行图像分析

        参数:
            image_input: 图像输入，支持以下格式：
                - str: Base64 编码字符串
                - np.ndarray: OpenCV BGR 格式
                - PIL.Image: PIL 图像
            prompt: 用户提问文本
            detection_summary: YOLO 检测结果摘要（注入 Prompt）
            retries: 重试次数

        返回:
            dict: {"success": bool, "answer": str, "error": str}
        """
        if not self._initialized:
            return {"success": False, "answer": "", "error": "DashScope 未初始化"}

        # 处理图像输入
        image_b64 = self._prepare_image(image_input)

        # 构造完整 Prompt
        full_prompt = self._build_prompt(prompt, detection_summary)

        # 构造 API 请求 messages
        messages = [{
            "role": "user",
            "content": [
                {"image": image_b64},
                {"text": full_prompt}
            ]
        }]

        for attempt in range(retries + 1):
            try:
                response = MultiModalConversation.call(model=self.model, messages=messages)
                if response.status_code == 200:
                    answer = response.output.choices[0].message.content[0]["text"]
                    return {"success": True, "answer": answer, "error": ""}
                else:
                    error_msg = f"API 返回错误: code={response.status_code}, msg={response.message}"
                    logger.warning(f"DashScope 调用失败 (尝试 {attempt + 1}): {error_msg}")
                    if attempt < retries:
                        time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"DashScope 调用异常 (尝试 {attempt + 1}): {e}")
                if attempt < retries:
                    time.sleep(2 ** attempt)
                else:
                    return {"success": False, "answer": "", "error": str(e)}

        return {"success": False, "answer": "", "error": "重试次数耗尽"}

    # ==================== 异步调用（非阻塞） ====================

    def analyze_async(
        self,
        image_input,
        prompt: str,
        callback: Callable[[dict], None],
        detection_summary: str = ""
    ) -> None:
        """
        异步调用 Qwen-VL（通过队列+工作线程）

        参数:
            image_input: 图像输入
            prompt: 用户提问
            callback: 结果回调函数 callback({"success": bool, "answer": str, "error": str})
            detection_summary: YOLO 检测结果摘要
        """
        if not self._initialized:
            callback({"success": False, "answer": "", "error": "DashScope 未初始化"})
            return

        # 预处理图像（在工作线程外完成，减少队列负担）
        image_b64 = self._prepare_image(image_input)
        full_prompt = self._build_prompt(prompt, detection_summary)

        try:
            self._task_queue.put_nowait({
                "image_b64": image_b64,
                "prompt": full_prompt,
                "callback": callback,
            })
        except Exception as e:
            logger.error(f"异步任务入队失败: {e}")
            callback({"success": False, "answer": "", "error": f"入队失败: {e}"})

        # 确保工作线程在运行
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        """确保后台工作线程在运行"""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """后台工作线程，消费队列中的 API 请求"""
        while self._worker_running:
            try:
                task = self._task_queue.get(timeout=1)
            except Empty:
                continue

            try:
                messages = [{
                    "role": "user",
                    "content": [
                        {"image": task["image_b64"]},
                        {"text": task["prompt"]}
                    ]
                }]

                response = MultiModalConversation.call(model=self.model, messages=messages)

                if response.status_code == 200:
                    answer = response.output.choices[0].message.content[0]["text"]
                    task["callback"]({"success": True, "answer": answer, "error": ""})
                else:
                    task["callback"]({
                        "success": False,
                        "answer": "",
                        "error": f"API 错误: {response.message}"
                    })
            except Exception as e:
                task["callback"]({
                    "success": False,
                    "answer": "",
                    "error": str(e)
                })
            finally:
                self._task_queue.task_done()

    def shutdown(self) -> None:
        """关闭后台工作线程"""
        self._worker_running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

    # ==================== 内部辅助方法 ====================

    def _prepare_image(self, image_input) -> str:
        """
        将各种图像输入格式统一为 Base64 字符串
        """
        import numpy as np

        if isinstance(image_input, str) and image_input.startswith("data:image"):
            # 已经是 Base64 编码
            return image_input

        if isinstance(image_input, np.ndarray):
            # OpenCV 帧 → 压缩 → Base64
            frame = compress_image_for_api(image_input, API_IMAGE_MAX_WIDTH)
            return frame_to_base64(frame)

        if isinstance(image_input, Image.Image):
            # PIL Image → Base64
            buffer = io.BytesIO()
            image_input.save(buffer, format="JPEG", quality=85)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"

        if isinstance(image_input, str) and os.path.exists(image_input):
            # 文件路径 → 读取 → Base64
            with open(image_input, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"

        raise ValueError(f"不支持的图像输入格式: {type(image_input)}")

    def _build_prompt(self, user_prompt: str, detection_summary: str = "") -> str:
        """
        构造完整的 Prompt，将 YOLO 检测结果作为上下文注入
        """
        if detection_summary:
            return (
                f"以下是 YOLO 目标检测的结果作为参考：\n"
                f"{detection_summary}\n\n"
                f"用户提问：{user_prompt}\n"
                f"请基于检测结果简洁准确地回答用户的问题。如果用户问的问题在检测结果中已有答案，"
                f"请直接使用检测结果回答；如果需要额外判断，请结合你的视觉理解回答。"
            )
        else:
            return f"请观察这张图片，回答以下问题：{user_prompt}"

    # ==================== 属性 ====================

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def queue_size(self) -> int:
        return self._task_queue.qsize()
