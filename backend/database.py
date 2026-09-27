"""
SQLite 数据库模块
持久化存储检测结果和事件记录，支持按时间查询和统计
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Optional

from backend.config import DATABASE_PATH


class DetectionDatabase:
    """检测结果数据库管理类"""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        # 确保数据库目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    # ==================== 数据库初始化 ====================

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # 提升并发写入性能
        return conn

    def _init_db(self) -> None:
        """初始化数据库表结构"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # 检测记录表：每一帧的 YOLO 检测结果
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,           -- ISO 格式时间戳
                class_name TEXT NOT NULL,          -- 检测类别名
                confidence REAL NOT NULL,          -- 置信度
                bbox_x1 REAL NOT NULL,             -- 边界框左上角 x
                bbox_y1 REAL NOT NULL,             -- 边界框左上角 y
                bbox_x2 REAL NOT NULL,             -- 边界框右下角 x
                bbox_y2 REAL NOT NULL,             -- 边界框右下角 y
                image_path TEXT,                   -- 快照图片路径
                event_type TEXT DEFAULT 'detection' -- 事件类型
            )
        """)

        # 事件记录表：汇总的事件（如"检测到人"）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,           -- 事件发生时间
                event_type TEXT NOT NULL,          -- 事件类型
                class_name TEXT,                   -- 涉及的类别
                description TEXT,                  -- 事件描述
                confidence REAL,                   -- 最高置信度
                snapshot_path TEXT,                -- 事件快照路径
                video_clip_path TEXT               -- 视频片段路径（预留）
            )
        """)

        # 为常用查询创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_detections_timestamp
            ON detections(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_detections_class
            ON detections(class_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp
            ON events(timestamp)
        """)

        conn.commit()
        conn.close()

    # ==================== 数据写入 ====================

    def save_detections(self, detections: list, image_path: str = None) -> None:
        """
        批量保存一帧中的所有检测结果
        detections: [{"class": "person", "confidence": 0.95, "bbox": [x1,y1,x2,y2]}, ...]
        """
        if not detections:
            return

        timestamp = datetime.now().isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()

        rows = [
            (
                timestamp,
                d["class"],
                d["confidence"],
                d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3],
                image_path,
                "detection"
            )
            for d in detections
        ]
        cursor.executemany(
            "INSERT INTO detections (timestamp, class_name, confidence, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2, image_path, event_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows
        )
        conn.commit()
        conn.close()

    def save_event(self, event_type: str, class_name: str, description: str,
                   confidence: float, snapshot_path: str = None) -> int:
        """
        保存一条事件记录
        返回事件 ID
        """
        timestamp = datetime.now().isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO events (timestamp, event_type, class_name, description, "
            "confidence, snapshot_path) VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp, event_type, class_name, description, confidence, snapshot_path)
        )
        event_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return event_id

    # ==================== 数据查询 ====================

    def get_daily_stats(self, date_str: str) -> list:
        """
        获取指定日期的各类别检测数量统计
        date_str: "YYYY-MM-DD"
        返回: [{"class_name": "person", "count": 150}, ...]
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT class_name, COUNT(*) as cnt FROM detections "
            "WHERE date(timestamp) = ? GROUP BY class_name ORDER BY cnt DESC",
            (date_str,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"class_name": r["class_name"], "count": r["cnt"]} for r in rows]

    def get_weekly_stats(self, start_date: str) -> list:
        """
        获取一周内的各类别检测数量统计
        start_date: "YYYY-MM-DD"（周一）
        """
        end_date = (datetime.fromisoformat(start_date) + timedelta(days=7)).strftime("%Y-%m-%d")
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT class_name, COUNT(*) as cnt FROM detections "
            "WHERE date(timestamp) >= ? AND date(timestamp) < ? "
            "GROUP BY class_name ORDER BY cnt DESC",
            (start_date, end_date)
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"class_name": r["class_name"], "count": r["cnt"]} for r in rows]

    def get_monthly_stats(self, year: int, month: int) -> list:
        """获取指定月份的各类别检测数量统计"""
        start = f"{year:04d}-{month:02d}-01"
        if month == 12:
            end = f"{year + 1:04d}-01-01"
        else:
            end = f"{year:04d}-{month + 1:02d}-01"
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT class_name, COUNT(*) as cnt FROM detections "
            "WHERE date(timestamp) >= ? AND date(timestamp) < ? "
            "GROUP BY class_name ORDER BY cnt DESC",
            (start, end)
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"class_name": r["class_name"], "count": r["cnt"]} for r in rows]

    def get_hourly_stats(self, date_str: str) -> list:
        """
        获取指定日期按小时分组的检测数量
        返回: [{"hour": 0, "count": 10}, ...]
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour, "
            "COUNT(*) as cnt FROM detections "
            "WHERE date(timestamp) = ? "
            "GROUP BY hour ORDER BY hour",
            (date_str,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"hour": r["hour"], "count": r["cnt"]} for r in rows]

    def get_events_by_timerange(self, start: str, end: str) -> list:
        """
        按时间范围查询事件记录
        start, end: ISO 格式时间戳或日期字符串
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM events WHERE timestamp >= ? AND timestamp <= ? "
            "ORDER BY timestamp DESC",
            (start, end)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_all_events(self, limit: int = 100) -> list:
        """获取最近的事件记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_detections_by_timerange(self, start: str, end: str,
                                    class_name: Optional[str] = None) -> list:
        """按时间范围和类别查询检测记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        if class_name:
            cursor.execute(
                "SELECT * FROM detections WHERE timestamp >= ? AND timestamp <= ? "
                "AND class_name = ? ORDER BY timestamp DESC",
                (start, end, class_name)
            )
        else:
            cursor.execute(
                "SELECT * FROM detections WHERE timestamp >= ? AND timestamp <= ? "
                "ORDER BY timestamp DESC",
                (start, end)
            )
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_available_dates(self) -> list:
        """获取数据库中有检测记录的所有日期"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT date(timestamp) as d FROM detections ORDER BY d DESC"
        )
        rows = cursor.fetchall()
        conn.close()
        return [r["d"] for r in rows]

    def get_confidence_distribution(self, date_str: str = None) -> list:
        """
        获取置信度分布数据（0-0.1, 0.1-0.2, ... 0.9-1.0）
        用于绘制直方图
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        if date_str:
            cursor.execute(
                "SELECT confidence FROM detections WHERE date(timestamp) = ?",
                (date_str,)
            )
        else:
            cursor.execute("SELECT confidence FROM detections")
        rows = cursor.fetchall()
        conn.close()

        # 统计各置信度区间
        bins = [0] * 10
        for r in rows:
            idx = min(int(r["confidence"] * 10), 9)
            bins[idx] += 1
        return [
            {"range": f"{i/10:.1f}-{(i+1)/10:.1f}", "count": bins[i]}
            for i in range(10)
        ]
