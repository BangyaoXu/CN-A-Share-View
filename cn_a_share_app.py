# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import time

st.set_page_config(layout="wide")
st.title("🇨🇳 CSI 300 T+1 主动交易系统 (Free API + 进度显示)")

# =============================
# iTick Free API Key
# =============================
API_TOKEN = st.secrets.get("ITICK_API_KEY")
if not API_TOKEN:
    st.error("请在 Streamlit Secrets 中配置 ITICK_API_KEY")
    st.stop()
HEADERS = {"accept": "application/json", "token": API_TOKEN}

CACHE_FILE = "csi300_cache.csv"

# =============================
# Hardcoded CSI300 components (partial demo, extend to full ~300)
# =============================
CSI300 = [
    {"symbol": "600519", "name": "贵州茅台", "region": "SH"},
    {"symbol": "000858", "name": "五粮液", "region": "SZ"},
    {"symbol": "601318", "name": "中国平安", "region": "SH"},
    {"symbol": "601166", "name": "兴业银行", "region": "SH"},
    {"symbol": "000333", "name": "美的集团", "region": "SZ"},
    # ... add all ~300 stocks
]
csi300_df = pd.DataFrame(CSI300)

# =============================
# API fetch functions
# =============================
def fetch_quote(region, code):
    url = f"https://api.itick.org/stock/quote?region={region}&code={code}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 200:
        return r.json().get("data", {})
    return {}

def fetch_stock_info(region, code):
    url = f"https://api.itick.org/stock/info?region={region}&code={code}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 200:
        return r.json().get("data", {})
    return {}

# =============================
# Fetch CSI300 quotes in batches with progress
# =============================
progress_placeholder = st.empty()
bar_placeholder = st.progress(0.0)

records = []
total_batches = (len(csi300_df) // 50) + 1

for i, start in enumerate(range(0, len(csi300_df), 50)):
    batch = csi300_df.iloc[start:start+50]
    for _, row in batch.iterrows():
        code = row["symbol"]
        region = row["region"]
        name = row["name"]
        quote = fetch_quote(region, code)
        info = fetch_stock_info(region, code)
        if not quote or not info:
            continue
        sector = info.get("i","其他板块")
        change = quote.get("change", 0)
        turnover = quote.get("turnover", 0)
        records.append({
            "代码": code,
            "名称": name,
            "板块": sector,
            "涨跌幅": change,
            "成交量": turnover
        })
    # update progress
    progress_placeholder.text(f"抓取 CSI300 数据: 批次 {i+1}/{total_batches}")
    bar_placeholder.progress((i+1)/total_batches)
    time.sleep(0.5)  # avoid hitting free API limits

df = pd.DataFrame(records)
if not df.empty:
    df.to_csv(CACHE_FILE, index=False)
progress_placeholder.text("CSI300 数据抓取完成！")
bar_placeholder.progress(1.0)

# =============================
# 板块动量打分
# =============================
sector_score = df.groupby("板块").agg({
    "涨跌幅":"mean",
    "成交量":"sum"
}).reset_index()
sector_score["热度"] = sector_score["涨跌幅"] + sector_score["成交量"]/1e6
sector_score = sector_score.sort_values("热度", ascending=False)
top_sectors = sector_score.head(10)

st.subheader("🔥 板块热度排行榜")
st.dataframe(top_sectors, use_container_width=True)

# =============================
# 板块龙头个股
# =============================
df["评分"] = df["涨跌幅"] + df["成交量"]/1e6
top_stocks = df.sort_values("评分", ascending=False).groupby("板块").head(3)

st.subheader("🔍 板块龙头个股")
st.dataframe(top_stocks[["板块","代码","名称","涨跌幅","成交量"]], use_container_width=True)

# =============================
# 风险评分
# =============================
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

# =============================
# 今日操作建议
# =============================
st.subheader("🎯 今日操作建议")
if total_score > 70:
    st.success("进攻模式：聚焦强势板块龙头，回踩买入")
elif total_score > 40:
    st.warning("精选模式：控制仓位，快进快出")
else:
    st.error("防守模式：降低仓位，避免追高")

st.caption(f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================
# 板块热力图
# =============================
fig = px.bar(top_sectors, x="板块", y="热度", color="热度",
             text="涨跌幅", title="板块热度排行榜")
st.plotly_chart(fig, use_container_width=True)
