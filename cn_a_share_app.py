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
import threading

st.set_page_config(layout="wide")
st.title("🇨🇳 A股 T+1 主动交易系统")

# ----------------------------
# 配置 API Key
# ----------------------------
API_TOKEN = st.secrets.get("ITICK_API_KEY")
if not API_TOKEN:
    st.error("请在 Streamlit Secrets 中配置 ITICK_API_KEY")
    st.stop()
HEADERS = {"accept": "application/json", "token": API_TOKEN}

CACHE_FILE = "stock_cache.csv"
PROGRESS_FILE = "progress.txt"

# ----------------------------
# 工具函数
# ----------------------------
def fetch_symbol_list(region):
    url = f"https://api.itick.org/symbol/list?type=stock&region={region}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 200:
        df = pd.DataFrame(r.json().get("data", []))
        df = df.rename(columns={"c":"symbol", "n":"name", "e":"region"})
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
# 全市场抓取函数（后台线程 + 进度写入）
# ----------------------------
def fetch_full_market_progress():
    sh_stocks = fetch_symbol_list("SH")
    sz_stocks = fetch_symbol_list("SZ")
    universe = pd.concat([sh_stocks, sz_stocks], ignore_index=True)
    total_batches = (len(universe) // 50) + 1
    records = []

    for i, start in enumerate(range(0, len(universe), 50)):
        batch = universe.iloc[start:start+50]
        for _, row in batch.iterrows():
            region = row["region"]
            code = row["symbol"]
            info = fetch_stock_info(region, code)
            quote = fetch_quote(region, code)
            if not info or not quote:
                continue
            name = info.get("n","")
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
        # 写入进度
        with open(PROGRESS_FILE,"w") as f:
            f.write(f"{i+1}/{total_batches}")
        time.sleep(1)  # 避免超限

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)
    # 完成后清除进度
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

# ----------------------------
# 加载缓存
# ----------------------------
@st.cache_data(ttl=86400)
def load_cached_data():
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)
    return pd.DataFrame()

df = load_cached_data()

# ----------------------------
# 启动后台线程抓取
# ----------------------------
threading.Thread(target=fetch_full_market_progress, daemon=True).start()

# ----------------------------
# 显示后台进度
# ----------------------------
if os.path.exists(PROGRESS_FILE):
    progress_text = st.empty()
    progress_bar = st.progress(0)
    def update_progress():
        while os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE,"r") as f:
                line = f.read()
            try:
                current, total = map(int,line.strip().split("/"))
                progress_bar.progress(current/total)
                progress_text.text(f"后台更新中: 批次 {current}/{total}")
            except:
                pass
            time.sleep(1)
    threading.Thread(target=update_progress, daemon=True).start()

# ----------------------------
# 如果缓存为空，提示用户
# ----------------------------
if df.empty:
    st.warning("全市场数据正在更新，请稍后刷新页面查看最新数据。")
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
