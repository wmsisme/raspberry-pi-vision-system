"""
YOLO 本地目标检测模块
加载 YOLO 模型，执行目标检测，返回结构化结果
支持 ONNX 格式模型和标准 PyTorch 模型
"""

import logging
import os
import time
from typing import Optional

import cv2
import numpy as np

from backend.config import (
    YOLO_MODEL_PATH, YOLO_MODEL_SHORT, YOLO_MODEL_NAME, YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD, YOLO_USE_ONNX, COCO_CLASSES, BASE_DIR
)

import shutil

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO 目标检测器，封装 Ultralytics YOLO 模型"""

    def __init__(
        self,
        model_path: str = YOLO_MODEL_PATH,
        conf_threshold: float = YOLO_CONF_THRESHOLD,
        iou_threshold: float = YOLO_IOU_THRESHOLD,
        use_onnx: bool = YOLO_USE_ONNX
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = COCO_CLASSES
        self._model = None
        self._model_loaded = False
        self._last_load_attempt = 0
        self._load_model()

    def _find_local_model(self, base_path: str) -> str:
        """
        在多个可能路径中搜索本地模型文件
        返回找到的路径，未找到返回空字符串
        """
        # .pt 和 .onnx 两种格式
        candidates = [base_path, base_path.replace(".pt", ".onnx")]

        # 也搜索 CWD 下的 models/ 目录
        cwd_path = os.path.join(os.getcwd(), "models", os.path.basename(base_path))
        candidates.append(cwd_path)
        candidates.append(cwd_path.replace(".pt", ".onnx"))

        # 也搜索 home 目录下的常见位置
        home_models = os.path.join(os.path.expanduser("~"), "models", os.path.basename(base_path))
        candidates.append(home_models)

        for path in candidates:
            if os.path.exists(path) and os.path.getsize(path) > 1000:  # 排除空/损坏文件
                logger.info(f"找到本地模型: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
                return path

        return ""

    def _load_model(self) -> None:
        """
        加载 YOLO 模型
        优先级: 多路径搜索本地文件 > ultralytics 自动下载（用短名称）
        自动下载后会复制到 models/ 目录作为本地缓存
        """
        now = time.time()
        if now - self._last_load_attempt < 5 and self._model_loaded:
            return  # 避免频繁重载
        self._last_load_attempt = now

        from ultralytics import YOLO

        # 打印诊断信息
        logger.info(f"BASE_DIR: {BASE_DIR}")
        logger.info(f"当前工作目录(CWD): {os.getcwd()}")
        logger.info(f"期望模型路径: {self.model_path}")
        logger.info(f"期望路径文件存在: {os.path.exists(self.model_path)}")

        # 1) 多路径搜索本地模型文件
        local_path = self._find_local_model(self.model_path)
        if local_path:
            logger.info(f"加载本地模型: {local_path}")
            try:
                self._model = YOLO(local_path)
                self._model_loaded = True
                self.model_path = local_path  # 更新为实际路径
                return
            except Exception as e:
                logger.warning(f"本地模型加载失败: {e}，尝试自动下载")

        # 2) 本地文件不存在 → 用短名称触发 ultralytics 自动下载
        logger.info(f"未在以下位置找到模型，使用短名称 '{YOLO_MODEL_SHORT}' 触发自动下载...")
        logger.info(f"  已搜索: {self.model_path}, CWD/models/, ~/models/")
        try:
            self._model = YOLO(YOLO_MODEL_SHORT)
            self._model_loaded = True
            logger.info(f"YOLO 模型自动下载并加载成功: {YOLO_MODEL_SHORT}")

            # 尝试将下载的模型复制到 models/ 目录作为本地缓存
            self._cache_downloaded_model()
            return
        except Exception as e:
            logger.error(f"YOLO 模型自动下载失败: {e}")
            logger.error(
                "请: 1) 确保模型文件在 models/ 目录下  2) 检查网络 "
                "3) 运行: ls -la models/  确认文件存在"
            )

        # 3) 所有方式都失败
        self._model_loaded = False
        logger.error("所有模型加载方式均失败，目标检测功能不可用")

    @staticmethod
    def _cache_search_dirs() -> list:
        """
        收集 ultralytics 可能存放下载权重的候选目录（跨平台）

        不能写死 ~/.cache/ultralytics —— 那是 Linux/macOS 的约定：
          * Linux   : ~/.cache/ultralytics
          * Windows : %APPDATA%\\Ultralytics  (settings['weights_dir'])
          * 树莓派  : ~/.cache/ultralytics
        另外 ultralytics 也可能把权重直接下到当时的 CWD。
        故按「settings 声明的权重目录 → 平台兼容路径 → CWD」依次收集。
        """
        dirs = []

        # 1) ultralytics 自己声明的权重目录（最权威，跨平台正确）
        try:
            from ultralytics import settings as ultra_settings
            wd = ultra_settings.get("weights_dir")
            if wd:
                dirs.append(os.path.expanduser(wd))
        except Exception:
            pass

        # 2) 平台兼容的常见缓存路径
        home = os.path.expanduser("~")
        dirs.append(os.path.join(home, ".cache", "ultralytics"))
        dirs.append(os.path.join(home, ".cache", "ultralytics", "hub", "checkpoints"))
        appdata = os.environ.get("APPDATA")
        if appdata:  # Windows
            dirs.append(os.path.join(appdata, "Ultralytics"))
            dirs.append(os.path.join(appdata, "Ultralytics", "weights"))
        dirs.append(os.path.join(home, "Library", "Caches", "ultralytics"))  # macOS
        dirs.append(os.path.join(home, ".config", "Ultralytics"))            # Linux XDG
        dirs.append(os.path.join(home, "Ultralytics"))

        # 3) 当时的 CWD 与项目 models/ 平级目录
        dirs.append(os.getcwd())
        dirs.append(os.path.join(BASE_DIR, "models"))

        # 去重 + 只保留真实存在的目录
        seen, out = set(), []
        for d in dirs:
            d = os.path.abspath(d)
            if d not in seen and os.path.isdir(d):
                seen.add(d)
                out.append(d)
        return out

    def _find_downloaded_weights(self) -> str:
        """在候选目录中查找刚下载的权重文件，返回其真实路径（未找到返回空串）"""
        targets = {YOLO_MODEL_NAME.lower(), f"{YOLO_MODEL_SHORT}.pt".lower()}
        for cache_dir in self._cache_search_dirs():
            if os.path.abspath(cache_dir) == os.path.abspath(os.path.dirname(self.model_path)):
                continue  # 跳过目标目录本身
            for root, _dirs, files in os.walk(cache_dir):
                for f in files:
                    if f.lower() in targets:
                        return os.path.join(root, f)
        return ""

    def _cache_downloaded_model(self) -> None:
        """
        把 ultralytics 自动下载的权重复制到本地 models/ 目录，并将
        self.model_path 更新为**磁盘上真实存在**的路径。

        重要: 只有在文件确实落盘后才更新 model_path —— 否则程序会对外
        报告一个并不存在的路径（本函数早期版本就有这个问题）。
        """
        try:
            # 已经就在目标位置，无需处理
            if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000:
                return

            src = self._find_downloaded_weights()
            if not src:
                logger.warning(
                    "未能定位 ultralytics 下载的权重文件，模型仅存在于内存中，"
                    "下次启动会重新下载。可手动把 %s 放入 models/ 目录",
                    YOLO_MODEL_NAME,
                )
                return

            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                shutil.copy2(src, self.model_path)
            except Exception as e:
                logger.warning(f"复制权重到 models/ 失败: {e}（不影响本次运行）")
                return

            if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000:
                # 文件确实落盘，self.model_path 保持指向它（= 磁盘上真实存在的路径）
                logger.info(f"已缓存模型到: {self.model_path} (来源: {src})")
            else:
                logger.warning(f"复制完成但目标文件不可用: {self.model_path}")
        except Exception as e:
            logger.warning(f"缓存模型异常: {e}")  # 缓存失败不影响运行

    def detect(self, frame: np.ndarray) -> list:
        """
        对单帧图像执行目标检测

        参数:
            frame: BGR 格式的 numpy array

        返回:
            list[dict]: 检测结果列表
            [
                {
                    "class": "person",       # 类别名称
                    "class_id": 0,           # 类别 ID
                    "confidence": 0.95,      # 置信度
                    "bbox": [x1, y1, x2, y2], # 边界框 (左上, 右下)
                    "center": [cx, cy],      # 边界框中心点 (归一化)
                    "area": 0.15,            # 边界框面积 (归一化)
                },
                ...
            ]
        """
        if not self._model_loaded:
            self._load_model()
            if not self._model_loaded:
                return []

        try:
            results = self._model(frame, verbose=False, conf=self.conf_threshold,
                                  iou=self.iou_threshold)
        except Exception as e:
            logger.error(f"YOLO 推理失败: {e}")
            return []

        if not results or len(results) == 0:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        h, w = frame.shape[:2]
        detections = []

        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = xyxy.tolist()

            # 计算中心点（归一化到 [0, 1]）
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h

            # 计算面积（归一化）
            area = ((x2 - x1) * (y2 - y1)) / (w * h)

            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

            detections.append({
                "class": class_name,
                "class_id": cls_id,
                "confidence": round(confidence, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "center": [round(cx, 3), round(cy, 3)],
                "area": round(area, 4),
            })

        return detections

    def draw_boxes(self, frame: np.ndarray, detections: list,
                   color_map: Optional[dict] = None) -> np.ndarray:
        """
        在图像上绘制检测边界框和标签

        参数:
            frame: 原始 BGR 图像
            detections: detect() 返回的检测结果列表
            color_map: 类别名到颜色的映射（可选）

        返回:
            np.ndarray: 标注后的图像
        """
        annotated = frame.copy()
        colors = {
            "person": (0, 255, 0),     # 绿色
            "car": (255, 0, 0),        # 蓝色
            "dog": (0, 165, 255),      # 橙色
            "cat": (255, 105, 180),    # 粉色
        }
        if color_map:
            colors.update(color_map)

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det["class"]
            conf = det["confidence"]
            color = colors.get(cls_name, (0, 255, 255))  # 默认黄色

            # 绘制边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def get_summary(self, detections: list) -> str:
        """
        生成检测结果的文本摘要，用于构造 DashScope Prompt

        参数:
            detections: detect() 返回的检测结果列表

        返回:
            str: 结构化文本摘要
        """
        if not detections:
            return "未检测到任何物体。"

        # 按类别分组统计
        from collections import Counter
        class_counts = Counter(d["class"] for d in detections)

        lines = [f"共检测到 {len(detections)} 个物体："]
        for cls_name, count in class_counts.most_common():
            # 找出该类别的最高置信度和边界框
            cls_dets = [d for d in detections if d["class"] == cls_name]
            max_conf = max(d["confidence"] for d in cls_dets)
            positions = []
            for d in cls_dets:
                cx, cy = d["center"]
                pos_desc = self._describe_position(cx, cy)
                positions.append(pos_desc)
            lines.append(f"- {cls_name}: {count} 个, 最高置信度 {max_conf:.2f}, "
                         f"位置: {', '.join(positions[:3])}")

        return "\n".join(lines)

    def _describe_position(self, cx: float, cy: float) -> str:
        """
        将归一化坐标转换为位置描述
        cx, cy: [0, 1] 归一化中心点坐标
        """
        horizontal = "左" if cx < 0.33 else ("右" if cx > 0.67 else "中")
        vertical = "上" if cy < 0.33 else ("下" if cy > 0.67 else "中")

        if horizontal == "中" and vertical == "中":
            return "中央"
        return f"{horizontal}{vertical}"

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    def reload_model(self, model_path: Optional[str] = None, conf: Optional[float] = None) -> bool:
        """
        重新加载模型（支持运行时切换模型或参数）

        参数:
            model_path: 新的模型路径（None 则不变）
            conf: 新的置信度阈值（None 则不变）
        """
        if model_path:
            self.model_path = model_path
        if conf is not None:
            self.conf_threshold = conf

        self._model = None
        self._model_loaded = False
        self._load_model()
        return self._model_loaded
