"""
页面4: 事件历史与回放
时间线界面，选择日期时间，查看和回放历史检测事件
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from PIL import Image

from backend.database import DetectionDatabase
from backend.config import init_session_state
from backend.logging_setup import setup_logging


@st.cache_resource
def get_database():
    """获取数据库实例"""
    try:
        return DetectionDatabase()
    except Exception as e:
        st.error(f"数据库连接失败: {e}")
        return None


def main():
    setup_logging()
    init_session_state()
    st.title("事件历史与回放")
    st.markdown("查看历史检测事件，回放检测快照")

    db = get_database()
    if db is None:
        st.info("数据库尚未初始化，请先在「实时检测」页面运行检测以生成数据")
        return

    # ==================== 时间筛选 ====================
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        available_dates = db.get_available_dates()
        if available_dates:
            selected_date = st.selectbox("选择日期", ["全部"] + available_dates)
        else:
            selected_date = "全部"

    with col2:
        all_events = db.get_all_events(limit=500)
        if all_events:
            all_classes = list(set(e.get("class_name", "") for e in all_events if e.get("class_name")))
            selected_class = st.selectbox(
                "筛选类别", ["全部"] + sorted(all_classes)
            )
        else:
            selected_class = "全部"

    with col3:
        min_conf = st.slider(
            "最低置信度", min_value=0.0, max_value=1.0, value=0.3, step=0.05
        )

    # ==================== 事件查询 ====================
    if selected_date == "全部":
        start = "2000-01-01T00:00:00"
        end = "2099-12-31T23:59:59"
    else:
        start = f"{selected_date}T00:00:00"
        end = f"{selected_date}T23:59:59"

    events = db.get_events_by_timerange(start, end)

    # 筛选
    if selected_class != "全部":
        events = [e for e in events if e.get("class_name") == selected_class]
    events = [e for e in events if e.get("confidence", 0) >= min_conf]

    # 同时查询检测记录（用于更精细的时间线展示）
    detections = db.get_detections_by_timerange(
        start, end,
        class_name=(selected_class if selected_class != "全部" else None)
    )

    # ==================== 概览统计 ====================
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("事件总数", len(events))
    with col_b:
        unique_classes = len(set(e.get("class_name", "") for e in events if e.get("class_name")))
        st.metric("涉及类别", unique_classes)
    with col_c:
        if events:
            avg_conf = sum(e.get("confidence", 0) for e in events) / len(events)
            st.metric("平均置信度", f"{avg_conf:.0%}")
        else:
            st.metric("平均置信度", "N/A")
    with col_d:
        st.metric("检测记录数", len(detections))

    # ==================== 事件时间线 ====================
    st.markdown("---")
    st.subheader("事件时间线")

    if not events:
        st.info("所选时间范围内暂无事件记录")
    else:
        # 按时间排序显示事件卡片
        for event in sorted(events, key=lambda e: e["timestamp"], reverse=True)[:50]:
            ts = event["timestamp"]
            if "T" in ts:
                display_time = ts.replace("T", " ")
            else:
                display_time = ts

            with st.container():
                col_time, col_info, col_img = st.columns([2, 4, 2])

                with col_time:
                    st.markdown(f"**{display_time[:19]}**")

                with col_info:
                    event_type = event.get("event_type", "detection")
                    class_name = event.get("class_name", "未知")
                    description = event.get("description", "")

                    # 事件类型标签颜色
                    color = "red" if event_type == "alert" else "blue"
                    st.markdown(
                        f":{color}[{event_type.upper()}] **{class_name}** "
                        f"置信度: {event.get('confidence', 0):.2f}"
                    )
                    if description:
                        st.caption(description)

                with col_img:
                    snapshot_path = event.get("snapshot_path", "")
                    if snapshot_path and os.path.exists(snapshot_path):
                        img = Image.open(snapshot_path)
                        st.image(img, width=120)
                    elif snapshot_path:
                        st.caption("快照文件不存在")

                st.divider()

    # ==================== 检测记录详细表格 ====================
    if detections:
        st.markdown("---")
        st.subheader("检测记录明细")

        df = pd.DataFrame(detections)
        if not df.empty:
            # 解析时间
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp", ascending=False)

            # 选择展示列
            display_cols = ["timestamp", "class_name", "confidence", "event_type"]
            available = [c for c in display_cols if c in df.columns]

            st.dataframe(
                df[available],
                width="stretch",
                hide_index=True,
                column_config={
                    "timestamp": "时间",
                    "class_name": "类别",
                    "confidence": st.column_config.NumberColumn("置信度", format="%.4f"),
                    "event_type": "事件类型",
                }
            )

            # 下载 CSV
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "导出 CSV",
                data=csv,
                file_name=f"detections_{selected_date}.csv",
                mime="text/csv",
                width="stretch"
            )


if __name__ == "__main__":
    main()
