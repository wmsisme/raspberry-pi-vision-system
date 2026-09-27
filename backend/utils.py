"""
工具函数模块
提供 Base64 编码、时间戳生成、告警播放、快照保存等通用功能
"""

import base64
import os
import time
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def frame_to_base64(frame: np.ndarray, ext: str = ".jpg") -> str:
    """
    将 OpenCV BGR 帧转换为 Base64 编码的 JPEG 字符串
    适用于 DashScope API 传输
    """
    # BGR → RGB → JPEG 编码
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def frame_to_bytes(frame: np.ndarray) -> bytes:
    """将 OpenCV BGR 帧转换为 JPEG 字节流"""
    _, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg_bytes.tobytes()


def generate_timestamp(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """生成当前时间的格式化字符串"""
    return datetime.now().strftime(fmt)


def generate_date_str() -> str:
    """生成日期字符串，用于按日期分子目录"""
    return datetime.now().strftime("%Y-%m-%d")


def ensure_dir(path: str) -> None:
    """确保目录存在，不存在则创建"""
    os.makedirs(path, exist_ok=True)


# 告警声通道 ID，与循环播放配套使用
_ALERT_CHANNEL_ID = 0


def _synthesize_beep(freq=800, duration_ms=300, sample_rate=22050):
    """合成一个简单的蜂鸣声波形（方波 + 指数衰减包络），返回 pygame Sound 对象"""
    import pygame
    import numpy as np
    import struct

    t = np.linspace(0, duration_ms / 1000.0, int(sample_rate * duration_ms / 1000.0), endpoint=False)
    # 方波模拟蜂鸣器效果（比正弦波更"吵"更响亮）+ 平缓衰减包络（让循环衔接更平滑）
    envelope = np.exp(-1.5 * t / (duration_ms / 1000.0))
    # 方波: sign(sin), 加入少量正弦谐波让声音更饱满
    square = np.sign(np.sin(2 * np.pi * freq * t))
    harmonic = 0.3 * np.sin(2 * np.pi * freq * 2 * t)  # 二次谐波增厚
    wave = ((square + harmonic) * envelope * 0.95).astype(np.float32)

    # 转为 16-bit PCM，满幅度输出
    samples = (wave * 32767).astype(np.int16)
    raw = struct.pack(f"<{len(samples)}h", *samples)
    return pygame.mixer.Sound(buffer=raw)


def _ensure_mixer():
    """确保 pygame mixer 已初始化并配置好"""
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=22050, size=-16, channels=2)
        pygame.mixer.set_num_channels(8)


def start_alert_sound(sound_path: str, duration_ms: int = 300) -> None:
    """
    开始循环播放告警声音，直到调用 stop_alert_sound()
    1) 优先使用指定的 .wav 文件
    2) 若 .wav 不存在，则合成蜂鸣声循环播放

    声音通过系统默认音频设备输出（USB 扬声器 / 3.5mm 接口均可）
    """
    try:
        import pygame
        _ensure_mixer()

        # 如果已经在播放，不重复启动
        channel = pygame.mixer.Channel(_ALERT_CHANNEL_ID)
        if channel.get_busy():
            return

        if os.path.exists(sound_path):
            sound = pygame.mixer.Sound(sound_path)
        else:
            sound = _synthesize_beep(freq=800, duration_ms=duration_ms)

        sound.set_volume(1.0)  # 最大音量
        channel.play(sound, loops=-1)  # -1 = 无限循环
    except Exception:
        pass  # 音频播放失败不影响主流程


def stop_alert_sound() -> None:
    """停止循环播放的告警声音"""
    try:
        import pygame
        if pygame.mixer.get_init():
            pygame.mixer.Channel(_ALERT_CHANNEL_ID).stop()
    except Exception:
        pass


def play_alert_sound(sound_path: str, duration_ms: int = 500) -> None:
    """
    播放一次告警声音（不循环），保留此函数以兼容其他调用方。
    如需持续告警，请使用 start_alert_sound / stop_alert_sound。
    """
    try:
        import pygame
        _ensure_mixer()

        if os.path.exists(sound_path):
            sound = pygame.mixer.Sound(sound_path)
        else:
            sound = _synthesize_beep(freq=800, duration_ms=duration_ms)

        sound.set_volume(1.0)
        sound.play()
    except Exception:
        pass


def imwrite_unicode(filepath: str, frame: np.ndarray, quality: int = 90) -> bool:
    """
    写入图像文件，正确处理**含非 ASCII 字符（如中文）的路径**。

    为什么不能直接用 cv2.imwrite:
        OpenCV 的 imwrite 在 Windows 上使用窄字符 API 打开文件，路径含中文时
        会**静默失败**——返回 False 且不创建文件，也不抛异常。本项目的根目录
        (`D:\\code_item\\树莓派`) 就含中文，会导致所有快照保存全部失效。

    做法: 先用 cv2.imencode 把图像编码到内存，再用 Python 的 open() 写文件
    （open() 走宽字符 API，支持任意 Unicode 路径）。

    返回 True 表示写入成功；失败时回退到 PIL 再试一次。
    """
    try:
        ensure_dir(os.path.dirname(filepath))
    except Exception:
        pass

    # 路径全 ASCII 时直接走 cv2 原生路径（性能更好）
    try:
        filepath.encode("ascii")
        if cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            return True
    except Exception:
        pass

    # 含非 ASCII 字符：编码到内存后由 Python 写文件
    try:
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            with open(filepath, "wb") as f:
                f.write(buffer.tobytes())
            return True
    except Exception:
        pass

    # 最后回退到 PIL
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(filepath, format="JPEG", quality=quality)
        return True
    except Exception:
        return False


def save_snapshot(frame: np.ndarray, base_dir: str, prefix: str = "snap") -> str:
    """
    保存检测快照到日期子目录
    返回保存的文件路径；保存失败返回空字符串（调用方需判空）
    """
    date_str = generate_date_str()
    save_dir = os.path.join(base_dir, date_str)

    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(save_dir, filename)

    if not imwrite_unicode(filepath, frame):
        import logging
        logging.getLogger(__name__).error(f"快照保存失败: {filepath}")
        return ""

    return filepath


def compress_image_for_api(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    """
    压缩图像以适应 API 传输
    保持宽高比缩放到 max_width 以内
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    new_h = int(h * max_width / w)
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """PIL Image → OpenCV BGR numpy array"""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """OpenCV BGR numpy array → PIL Image"""
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
