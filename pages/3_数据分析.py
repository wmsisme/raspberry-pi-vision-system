"""
页面3: 数据分析
日/周/月检测统计图表，置信度分布，事件类型占比
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.database import DetectionDatabase
from backend.config import init_session_state
from backend.logging_setup import setup_logging


@st.cache_resource
def get_database():
    """获取数据库实例（缓存）"""
    try:
        return DetectionDatabase()
    except Exception as e:
        st.error(f"数据库连接失败: {e}")
        return None


def main():
    setup_logging()
    init_session_state()
    st.title("数据统计与分析")
    st.markdown("检测结果的日/周/月统计图表")

    db = get_database()
    if db is None:
        st.info("数据库尚未初始化，请先在「实时检测」页面运行检测以生成数据")
        return

    # ==================== 时间范围选择 ====================
    col1, col2 = st.columns([1, 3])

    with col1:
        range_type = st.radio("时间范围", ["日", "周", "月"], horizontal=True)

    available_dates = db.get_available_dates()

    with col2:
        if range_type == "日":
            if available_dates:
                selected_date = st.selectbox("选择日期", available_dates)
            else:
                selected_date = datetime.now().strftime("%Y-%m-%d")
                st.info("暂无检测数据")
            stats = db.get_daily_stats(selected_date)
            hourly = db.get_hourly_stats(selected_date)
            conf_dist = db.get_confidence_distribution(selected_date)
        elif range_type == "周":
            if available_dates:
                selected_date = st.selectbox(
                    "选择周一日期",
                    available_dates,
                    index=0
                )
            else:
                selected_date = datetime.now().strftime("%Y-%m-%d")
                st.info("暂无检测数据")
            stats = db.get_weekly_stats(selected_date)
            hourly = []
            conf_dist = db.get_confidence_distribution()
        else:  # 月
            today = datetime.now()
            year = st.selectbox("年份", range(today.year, today.year - 3, -1))
            month = st.selectbox("月份", range(1, 13), index=today.month - 1)
            stats = db.get_monthly_stats(year, month)
            hourly = []
            conf_dist = db.get_confidence_distribution()
            selected_date = f"{year}-{month:02d}"

    # ==================== 统计图表 ====================
    if not stats:
        st.warning("所选时间范围内暂无检测数据")
        return

    st.markdown("---")

    # 第一行：类别分布柱状图 + 置信度分布
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader(f"各类别检测数量 ({selected_date})")
        df_stats = pd.DataFrame(stats)
        if not df_stats.empty:
            fig = px.bar(
                df_stats, x="class_name", y="count",
                color="count", color_continuous_scale="Viridis",
                labels={"class_name": "类别", "count": "检测次数"},
                height=400
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

    with col_chart2:
        st.subheader("置信度分布")
        df_conf = pd.DataFrame(conf_dist)
        if not df_conf.empty:
            fig = px.bar(
                df_conf, x="range", y="count",
                labels={"range": "置信度区间", "count": "数量"},
                height=400, color_discrete_sequence=["#636EFA"]
            )
            st.plotly_chart(fig, width="stretch")

    # 第二行：按小时趋势（仅日视图） + 类别占比饼图
    if hourly:
        col_chart3, col_chart4 = st.columns(2)

        with col_chart3:
            st.subheader("按小时检测趋势")
            df_hourly = pd.DataFrame(hourly)
            if not df_hourly.empty:
                # 补齐 0-23 小时
                all_hours = pd.DataFrame({"hour": range(24)})
                df_hourly = all_hours.merge(df_hourly, on="hour", how="left").fillna(0)
                fig = px.line(
                    df_hourly, x="hour", y="count",
                    markers=True,
                    labels={"hour": "小时", "count": "检测次数"},
                    height=350
                )
                fig.update_xaxes(dtick=1)
                st.plotly_chart(fig, width="stretch")

        with col_chart4:
            st.subheader("类别占比")
            if not df_stats.empty:
                fig = px.pie(
                    df_stats, names="class_name", values="count",
                    height=350
                )
                st.plotly_chart(fig, width="stretch")

    # ==================== 事件记录 ====================
    st.markdown("---")
    st.subheader("最近事件记录")

    events = db.get_all_events(limit=50)
    if events:
        df_events = pd.DataFrame(events)
        df_events["timestamp"] = pd.to_datetime(df_events["timestamp"])
        df_events = df_events.sort_values("timestamp", ascending=False)
        st.dataframe(
            df_events[["timestamp", "event_type", "class_name", "description", "confidence"]],
            width="stretch",
            hide_index=True,
            column_config={
                "timestamp": "时间",
                "event_type": "事件类型",
                "class_name": "类别",
                "description": "描述",
                "confidence": st.column_config.NumberColumn("置信度", format="%.2f")
            }
        )
    else:
        st.info("暂无事件记录")


if __name__ == "__main__":
    main()
