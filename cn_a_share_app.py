# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import time
import os

st.set_page_config(layout="wide")
st.title("🇨🇳 CSI 300 T+1 主动交易系统 (iTick Free API + 进度显示)")

# ----------------------------
# iTick API Key (set in Streamlit Secrets)
# ----------------------------
API_TOKEN = st.secrets.get("ITICK_API_KEY")
if not API_TOKEN:
    st.error("请在 Streamlit Secrets 中配置 ITICK_API_KEY")
    st.stop()

HEADERS = {"accept": "application/json", "token": API_TOKEN}
CACHE_FILE = "csi300_cache.csv"

# ----------------------------
# 工具函数
# ----------------------------
def fetch_csi300_components():
    """获取 CSI300 成分股"""
    url = "https://api.itick.org/index/component?region=CN&code=000300"
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 200:
        data = r.json().get("data", [])
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.rename(columns={"c":"symbol","n":"name","e":"region"})
        return df
    return pd.DataFrame()

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

# ----------------------------
# 全市场抓取函数（CSI300）+进度
# ----------------------------
def fetch_csi300_full(progress_placeholder, bar_placeholder):
    df_components = fetch_csi300_components()
    if df_components.empty:
        st.error("无法获取 CSI300 成分股，请检查 API Key 或网络")
        return pd.DataFrame()

    total_batches = (len(df_components) // 50) + 1
    records = []

    for i, start in enumerate(range(0, len(df_components), 50)):
        batch = df_components.iloc[start:start+50]
        for _, row in batch.iterrows():
            code = row["symbol"]
            region = row["region"] if "region" in row else ("SH" if code.startswith("6") else "SZ")
            info = fetch_stock_info(region, code)
            quote = fetch_quote(region, code)
            if not info or not quote:
                continue
            name = row["name"]
            sector = info.get("i","其他板块")
            change = quote.get("change",0)
            turnover = quote.get("turnover",0)
            records.append({
                "代码": code,
                "名称": name,
                "板块": sector,
                "涨跌幅": change,
                "成交量": turnover
            })
        # 更新进度
        progress = (i+1)/total_batches
        progress_placeholder.text(f"抓取 CSI300 数据: 批次 {i+1}/{total_batches}")
        bar_placeholder.progress(progress)
        time.sleep(1)  # 避免免费 API 限制

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)
    progress_placeholder.text("CSI300 数据抓取完成！")
    bar_placeholder.progress(1.0)
    return df

# ----------------------------
# 加载缓存
# ----------------------------
if os.path.exists(CACHE_FILE):
    df = pd.read_csv(CACHE_FILE)
    st.success(f"加载缓存数据，共 {len(df)} 条股票记录")
else:
    df = pd.DataFrame()

progress_placeholder = st.empty()
bar_placeholder = st.progress(0.0)

if df.empty:
    # 如果缓存为空，必须等待抓取
    df = fetch_csi300_full(progress_placeholder, bar_placeholder)
else:
    # 可选：后台更新
    st.info("后台正在更新 CSI300 数据…")
    import threading
    threading.Thread(target=fetch_csi300_full, args=(progress_placeholder, bar_placeholder), daemon=True).start()

if df.empty:
    st.warning("CSI300 数据仍在更新，请稍后刷新页面。")
    st.stop()

# ----------------------------
# 板块动量打分
# ----------------------------
sector_score = df.groupby("板块").agg({
    "涨跌幅":"mean",
    "成交量":"sum"
}).reset_index()
sector_score["热度"] = sector_score["涨跌幅"] + sector_score["成交量"]/1e6
sector_score = sector_score.sort_values("热度", ascending=False)
top_sectors = sector_score.head(10)

st.subheader("🔥 板块热度排行榜")
st.dataframe(top_sectors, use_container_width=True)

# ----------------------------
# 板块龙头个股
# ----------------------------
df["评分"] = df["涨跌幅"] + df["成交量"]/1e6
top_stocks = df.sort_values("评分", ascending=False).groupby("板块").head(3)

st.subheader("🔍 板块龙头个股")
st.dataframe(top_stocks[["板块","代码","名称","涨跌幅","成交量"]], use_container_width=True)

# ----------------------------
# 风险评分
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
