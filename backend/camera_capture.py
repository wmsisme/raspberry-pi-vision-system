"""
摄像头采集模块
统一摄像头接口，支持 CSI（picamera2）和 USB（OpenCV）摄像头
使用独立线程持续读取，减少主线程阻塞
"""

import atexit
import os
import signal
import subprocess
import threading
import time
import logging
from typing import Optional

import cv2
import numpy as np

from backend.config import (
    CAMERA_DEVICE_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    CAMERA_USE_PICAMERA2, CAMERA_RETRY_COUNT, CAMERA_RETRY_INTERVAL
)

logger = logging.getLogger(__name__)


class CameraCapture:
    """摄像头采集类，支持 USB 和 CSI 摄像头"""

    def __init__(
        self,
        device_id: int = CAMERA_DEVICE_ID,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
        use_picamera2: bool = CAMERA_USE_PICAMERA2
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._picam2 = None  # picamera2 实例
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.use_picamera2 = use_picamera2
        self._released = False  # 防止重复释放

        self._initialize()

        # 双重保障：进程退出时释放摄像头（atexit + signal）
        atexit.register(self.release)
        self._register_signal_handlers()

    def _register_signal_handlers(self):
        """注册 SIGTERM/SIGINT 信号处理器，确保强杀时也能释放资源"""
        capture = self  # 闭包捕获当前实例引用

        def _signal_handler(signum, frame):
            logger.info("收到终止信号，释放摄像头资源...")
            # 1) 释放当前实例的摄像头资源（OpenCV / picamera2）
            capture.release()
            # 2) 强制清理 libcamera 全局管线资源
            CameraCapture._force_release_picamera2_global()
            # 3) 引发 KeyboardInterrupt 让 Streamlit 正常退出，进而触发 atexit
            raise KeyboardInterrupt

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _signal_handler)
            except Exception:
                pass  # 非主线程注册 signal 可能失败，忽略

    def _initialize(self) -> None:
        """初始化摄像头"""
        if self.use_picamera2:
            try:
                self._init_picamera2()
            except RuntimeError as e:
                logger.warning(f"{e}")
                logger.warning("回退到 OpenCV USB 摄像头")
                self.use_picamera2 = False
                self._init_opencv()
        else:
            self._init_opencv()

    @staticmethod
    def _force_release_picamera2_global() -> None:
        """尝试强制释放 libcamera 全局资源（清理残留管线）

        libcamera 的 "Pipeline handler in use" 错误源于内核级管线锁，
        单靠 Python API 无法释放。本方法组合多种策略：
        1) 干掉残留的 libcamera 系统进程
        2) 通过 picamera2 API 重置 CameraManager
        3) 创建临时实例触发内部清理
        """
        # ---- 策略1: 干掉系统中残留的 libcamera 进程 ----
        # 这些进程可能是上次 Streamlit 异常退出后残留的 libcamerify 或 v4l2 子进程
        for target in ("libcamera", "libcamerify", "v4l2"):
            try:
                subprocess.run(
                    ["pkill", "-9", "-f", target],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3
                )
            except Exception:
                pass

        # ---- 策略2: 释放 /dev/video 设备上的文件锁 ----
        try:
            subprocess.run(
                ["fuser", "-k", "/dev/video0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3
            )
        except Exception:
            pass

        # ---- 策略3: picamera2 API 清理 ----
        try:
            from picamera2 import Picamera2
            Picamera2.set_logging(False)
            # 尝试通过 global_camera_manager 清理
            try:
                cm = Picamera2.global_camera_manager()
                cameras = getattr(cm, 'cameras', [])
                for cam in cameras:
                    logger.info(f"Force-releasing camera: {cam}")
                if hasattr(cm, 'setup'):
                    cm.setup()
            except Exception:
                pass

            # 创建临时实例再关闭，触发 libcamera 内部清理
            try:
                tmp = Picamera2()
                tmp.close()
            except Exception:
                pass
        except Exception:
            pass

    def _init_picamera2(self) -> None:
        """初始化 CSI 摄像头（通过 picamera2）"""
        from picamera2 import Picamera2

        # ---- 预清理：释放上次残留的管线锁 ----
        self._force_release_picamera2_global()

        last_error = None
        for attempt in range(CAMERA_RETRY_COUNT):
            try:
                self._picam2 = Picamera2()
                config = self._picam2.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self._picam2.configure(config)
                self._picam2.start()
                logger.info("Picamera2 (CSI) 摄像头初始化成功")
                return
            except ImportError:
                logger.warning("未安装 picamera2，回退到 OpenCV USB 摄像头")
                self.use_picamera2 = False
                self._init_opencv()
                return
            except Exception as e:
                last_error = e
                # 清理本实例的半初始化对象
                if self._picam2:
                    try:
                        self._picam2.close()
                    except Exception:
                        pass
                    self._picam2 = None

                logger.warning(
                    f"Picamera2 初始化失败 (尝试 {attempt + 1}/{CAMERA_RETRY_COUNT}): {e}"
                )
                if attempt < CAMERA_RETRY_COUNT - 1:
                    # 尝试强制释放全局管线资源，然后等待后重试
                    self._force_release_picamera2_global()
                    time.sleep(CAMERA_RETRY_INTERVAL)

        # 所有重试均失败，抛异常让调用方决定如何处理（不再静默回退）
        raise RuntimeError(
            f"Picamera2 初始化失败（已重试 {CAMERA_RETRY_COUNT} 次）: {last_error}"
        )

    def _init_opencv(self) -> None:
        """初始化 USB 摄像头（通过 OpenCV），支持自动重连"""
        # ---- 预清理：释放可能残留的摄像头句柄 ----
        # 场景：上次进程被强杀导致摄像头未被正常释放，
        # 通过临时创建 VideoCapture 再立即释放来"抢"回设备控制权
        try:
            stale_cap = cv2.VideoCapture(self.device_id)
            if stale_cap.isOpened():
                logger.info(f"检测到摄像头 {self.device_id} 残留句柄，尝试释放...")
            stale_cap.release()
        except Exception:
            pass

        for attempt in range(CAMERA_RETRY_COUNT):
            try:
                self._cap = cv2.VideoCapture(self.device_id)
                if self._cap.isOpened():
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self._cap.set(cv2.CAP_PROP_FPS, self.fps)
                    logger.info(f"OpenCV USB 摄像头初始化成功 (device={self.device_id})")
                    return
                else:
                    logger.warning(f"摄像头 {self.device_id} 打开失败，重试 {attempt + 1}/{CAMERA_RETRY_COUNT}")
            except Exception as e:
                logger.error(f"OpenCV 初始化异常: {e}")
            time.sleep(CAMERA_RETRY_INTERVAL)

        raise RuntimeError(f"无法打开摄像头设备 {self.device_id}")

    def start(self) -> None:
        """启动后台采集线程

        注意: release() 会置位 _released 闩锁，stop() 之后再 start() 必须复位它，
        否则后续 release() 会直接返回、摄像头句柄永不释放（设备被占死）。
        """
        if self._running:
            return
        self._released = False  # 复位释放闩锁，允许再次释放
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("摄像头采集线程已启动")

    def stop(self) -> None:
        """停止后台采集线程"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self.release()
        logger.info("摄像头采集线程已停止")

    def _capture_loop(self) -> None:
        """后台采集循环"""
        frame_interval = 1.0 / max(self.fps, 1)

        while self._running:
            frame = self._read_frame()
            if frame is not None:
                with self._lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.1)  # 读取失败时短暂等待
                continue
            time.sleep(frame_interval)

    def _read_frame(self) -> Optional[np.ndarray]:
        """从摄像头读取一帧"""
        if self.use_picamera2 and self._picam2:
            try:
                frame = self._picam2.capture_array()
                if frame is not None:
                    # picamera2 返回 RGB，转为 OpenCV BGR
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.error(f"Picamera2 读取帧失败: {e}")
            # picamera2 失败时走重连逻辑，不继续检查 _cap
            self._reconnect()

        elif self._cap:
            ret, frame = self._cap.read()
            if ret:
                return frame
            # OpenCV 读取失败，走重连
            self._reconnect()

        return None

    _reconnect_consecutive = 0   # 类级别计数器，避免日志刷屏
    _last_reconnect_time = 0.0   # 类级别冷却时间戳

    def _reconnect(self) -> None:
        """摄像头重连（支持 picamera2 和 OpenCV），带冷却避免死循环"""
        now = time.time()

        # ---- 冷却机制：两次重连之间至少间隔 5 秒 ----
        if now - CameraCapture._last_reconnect_time < 5.0:
            return
        CameraCapture._last_reconnect_time = now

        CameraCapture._reconnect_consecutive += 1

        if (CameraCapture._reconnect_consecutive <= 1
                or CameraCapture._reconnect_consecutive % 10 == 0):
            logger.warning(
                f"摄像头读取失败，尝试重连... "
                f"(第 {CameraCapture._reconnect_consecutive} 次)"
            )

        if self.use_picamera2:
            # ---- picamera2 重连 ----
            if self._picam2:
                try:
                    self._picam2.stop()
                except Exception:
                    pass
                try:
                    self._picam2.close()
                except Exception:
                    pass
                self._picam2 = None
            self._force_release_picamera2_global()
            try:
                self._init_picamera2()
                CameraCapture._reconnect_consecutive = 0
                logger.info("Picamera2 重连成功")
                return
            except Exception as e:
                logger.warning(f"Picamera2 重连失败: {e}")
                logger.warning("尝试回退到 OpenCV 模式...")
                self.use_picamera2 = False
                # 清理后再试 OpenCV
                self._force_release_picamera2_global()
                time.sleep(1)

        # ---- OpenCV 重连 / 回退 ----
        if self._cap:
            self._cap.release()
            self._cap = None
        try:
            self._init_opencv()
            CameraCapture._reconnect_consecutive = 0
            logger.info("OpenCV 摄像头重连成功")
        except Exception as e:
            logger.error(f"OpenCV 摄像头重连失败: {e}")
            # 如果 OpenCV 也失败了 10 次以上，尝试切回 picamera2
            if CameraCapture._reconnect_consecutive >= 10:
                logger.warning("OpenCV 持续失败，切回 picamera2 再试...")
                self.use_picamera2 = True

    def get_frame(self) -> Optional[np.ndarray]:
        """
        获取最新帧（线程安全）
        返回 BGR 格式的 numpy array，无帧时返回 None
        """
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None

    def release(self) -> None:
        """释放所有资源（可多次安全调用）"""
        if self._released:
            return
        self._released = True

        if self._cap:
            self._cap.release()
            self._cap = None
        if self._picam2:
            try:
                self._picam2.stop()
            except Exception:
                pass
            try:
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # 无 GUI 环境（如树莓派 headless）会抛异常，忽略即可
        logger.info("摄像头资源已释放")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def has_frame(self) -> bool:
        with self._lock:
            return self._latest_frame is not None
