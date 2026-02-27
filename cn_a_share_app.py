# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(layout="wide")
st.title("🇨🇳 CSI 300 T+1 主动交易系统 (CSV + 测试数据)")

# ----------------------------
# Load CSV
# ----------------------------
CSV_FILE = "csi300_full.csv"
try:
    csi300_df = pd.read_csv(CSV_FILE)
except Exception as e:
    st.error(f"无法读取 {CSV_FILE}：{e}")
    st.stop()

# ----------------------------
# Progress bar placeholder
# ----------------------------
progress_placeholder = st.empty()
bar_placeholder = st.progress(0.0)

# ----------------------------
# Simulate quotes for testing
# ----------------------------
records = []
total = len(csi300_df)
for i, row in enumerate(csi300_df.itertuples()):
    code = row.symbol
    name = row.name
    region = row.region
    # Simulated data
    change = round(np.random.uniform(-3, 3), 2)
    turnover = int(np.random.uniform(1000, 100000))
    sector = f"板块{np.random.randint(1,5)}"  # simulate 4 sectors
    records.append({
        "代码": code,
        "名称": name,
        "板块": sector,
        "涨跌幅": change,
        "成交量": turnover
    })
    # update progress
    progress_placeholder.text(f"抓取 CSI300 数据: {i+1}/{total}")
    bar_placeholder.progress((i+1)/total)
    time.sleep(0.05)  # small delay to show progress

df = pd.DataFrame(records)
progress_placeholder.text("CSI300 数据抓取完成！")
bar_placeholder.progress(1.0)

# ----------------------------
# 板块热度排行榜
# ----------------------------
sector_score = df.groupby("板块").agg({
    "涨跌幅":"mean",
    "成交量":"sum"
}).reset_index()
sector_score["热度"] = sector_score["涨跌幅"] + sector_score["成交量"]/1e5
sector_score = sector_score.sort_values("热度", ascending=False)
top_sectors = sector_score.head(10)

st.subheader("🔥 板块热度排行榜")
st.dataframe(top_sectors, use_container_width=True)

# ----------------------------
# 板块龙头个股
# ----------------------------
df["评分"] = df["涨跌幅"] + df["成交量"]/1e5
top_stocks = df.sort_values("评分", ascending=False).groupby("板块").head(3)

st.subheader("🔍 板块龙头个股")
st.dataframe(top_stocks[["板块","代码","名称","涨跌幅","成交量"]], use_container_width=True)

# ----------------------------
# 综合评分
# ----------------------------
macro_score = 50
liquidity_score = 50
sentiment_score = min(len(top_stocks),100)
total_score = np.mean([macro_score, liquidity_score, sentiment_score])

def gauge(title, value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={'axis': {'range':[0,100]}}
    ))
    fig.update_layout(height=250)
    return fig

st.subheader("📊 综合评分")
col1,col2,col3 = st.columns(3)
col1.plotly_chart(gauge("宏观评分", macro_score))
col2.plotly_chart(gauge("流动性评分", liquidity_score))
col3.plotly_chart(gauge("情绪评分", sentiment_score))
st.markdown(f"## 🔥 综合评分: {round(total_score,1)}")

# ----------------------------
# 今日操作建议
# ----------------------------
st.subheader("🎯 今日操作建议")
if total_score > 70:
    st.success("进攻模式：聚焦强势板块龙头，回踩买入")
elif total_score > 40:
    st.warning("精选模式：控制仓位，快进快出")
else:
    st.error("防守模式：降低仓位，避免追高")

st.caption(f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ----------------------------
# 板块热力图
# ----------------------------
fig = px.bar(top_sectors, x="板块", y="热度", color="热度",
             text="涨跌幅", title="板块热度排行榜")
st.plotly_chart(fig, use_container_width=True)
