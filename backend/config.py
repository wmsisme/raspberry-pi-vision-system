"""
全局配置文件
集中管理所有配置项，避免硬编码
"""

import os

# ==================== 项目根路径 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== .env 文件加载（优先级最高） ====================
def _load_dotenv():
    """尝试从项目根目录的 .env 文件加载环境变量"""
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except Exception:
            pass  # .env 读取失败不影响启动

_load_dotenv()

# ==================== DashScope API 配置 ====================
def get_dashscope_api_key() -> str:
    """
    懒加载 API Key，按优先级查找:
    1) 环境变量 DASHSCOPE_API_KEY
    2) Streamlit secrets (DASHSCOPE_API_KEY 字段)
    3) .env 文件 (已在 _load_dotenv 中加载到 os.environ)
    4) 默认占位符
    """
    # 1) 环境变量
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if key and key != "your-api-key-here":
        return key

    # 2) Streamlit secrets (运行时可用)
    try:
        import streamlit as st
        key = st.secrets.get("DASHSCOPE_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    # 3) 兜底
    return os.environ.get("DASHSCOPE_API_KEY", "your-api-key-here")

# 注意: 不要在此处固化模块级常量来保存 API Key。
# 模块级常量在 import 时求值一次，用户在 Streamlit 侧边栏输入的 Key 只写入
# os.environ，常量会一直是旧的占位符，造成"明明填了却提示未配置"。
# 需要 Key 时一律调用 get_dashscope_api_key()（每次读取，始终最新）。
# 模型选择: qwen-vl-plus（性价比） 或 qwen-vl-max（最强）
DASHSCOPE_MODEL = "qwen-vl-plus"

# ==================== YOLO 模型配置 ====================
# 模型短名称（ultralytics 官方自动下载标识，如 yolo11n / yolo11s / yolo8n 等）
# 完整列表: https://docs.ultralytics.com/models/
YOLO_MODEL_SHORT = "yolo11n"
# 模型文件名（放在 models/ 目录下，作为本地缓存）
YOLO_MODEL_NAME = f"{YOLO_MODEL_SHORT}.pt"
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", YOLO_MODEL_NAME)
# 置信度阈值（低于此值的检测结果将被过滤）
YOLO_CONF_THRESHOLD = 0.35
# NMS 的 IoU 阈值
YOLO_IOU_THRESHOLD = 0.45
# 是否使用 ONNX 格式模型（优先检测 models/ 下同名 .onnx 文件）
YOLO_USE_ONNX = False

# ==================== 摄像头配置 ====================
# 摄像头设备 ID（USB 摄像头通常为 0；CSI 摄像头使用 picamera2 时此值忽略）
CAMERA_DEVICE_ID = 0
# 目标分辨率
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# 目标帧率
CAMERA_FPS = 15
# 是否优先使用 picamera2（树莓派 CSI 摄像头）
CAMERA_USE_PICAMERA2 = False
# 摄像头断连重试次数
CAMERA_RETRY_COUNT = 5
# 摄像头断连重试间隔（秒）
CAMERA_RETRY_INTERVAL = 2

# ==================== 数据库配置 ====================
DATABASE_PATH = os.path.join(BASE_DIR, "data", "detections.db")

# ==================== 快照存储配置 ====================
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")

# ==================== 日志配置 ====================
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ==================== 告警配置 ====================
# 告警声音文件路径
ALERT_SOUND_PATH = os.path.join(BASE_DIR, "assets", "alert.wav")
# 默认告警目标类别（可通过 Streamlit 侧边栏修改）
ALERT_CLASSES = ["person", "dog", "cat"]

# ==================== 事件检测配置 ====================
# 连续多少帧检测到同一事件才触发记录（避免误报）
EVENT_CONSECUTIVE_FRAMES = 3
# 事件记录最小间隔（秒），同一事件在此时间内不重复记录
EVENT_MIN_INTERVAL = 5

# ==================== Streamlit 页面配置 ====================
STREAMLIT_PAGE_TITLE = "树莓派智能视觉系统"
STREAMLIT_PAGE_ICON = "📷"
STREAMLIT_LAYOUT = "wide"

# ==================== API 调用配置 ====================
# 异步 API 调用超时（秒）
API_TIMEOUT = 30
# 最大重试次数
API_MAX_RETRIES = 2
# API 调用前图片压缩最大宽度（像素）
API_IMAGE_MAX_WIDTH = 640

# ==================== COCO 数据集类别名（80类） ====================
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# ==================== Session State 初始化 ====================
def init_session_state():
    """
    初始化 Streamlit session_state 中的全局对象
    每个页面在 main() 开头调用，确保直接访问子页面时也不会报 AttributeError
    """
    import streamlit as st

    defaults = {
        "camera": None,               # 摄像头实例（懒加载）
        "yolo_detector": None,         # YOLO 检测器实例
        "database": None,             # 数据库实例
        "dashscope_client": None,     # DashScope 客户端实例
        "detection_running": False,    # 检测运行状态
        "current_frame": None,        # 当前帧
        "latest_detections": [],      # 最新检测结果
        "annotated_frame": None,      # 标注后的帧
        "chat_history": [],           # 聊天历史（拍照问答页）
        "qa_image": None,             # 当前上传/拍摄的图像
        "qa_detections": [],          # 当前图像的检测结果
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
