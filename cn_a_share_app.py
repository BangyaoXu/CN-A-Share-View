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
st.title("🇨🇳 A股 T+1 主动交易系统 (iTick Free API 云端缓存版)")

# ----------------------------
# 配置 API Key
# ----------------------------
API_TOKEN = st.secrets.get("ITICK_API_KEY")
if not API_TOKEN:
    st.error("请在 Streamlit Secrets 中配置 ITICK_API_KEY")
    st.stop()

HEADERS = {"accept": "application/json", "token": API_TOKEN}

# ----------------------------
# 缓存文件路径
# ----------------------------
CACHE_FILE = "stock_cache.csv"

# ----------------------------
# 工具函数
# ----------------------------
def fetch_symbol_list(region):
    url = f"https://api.itick.org/symbol/list?type=stock&region={region}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 200:
        df = pd.DataFrame(r.json().get("data", []))
        # 重命名列，方便后续使用
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
# 数据抓取/缓存逻辑
# ----------------------------
@st.cache_data(ttl=86400)  # 每天刷新一次缓存
def load_data():
    if os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE)
        return df
    # 如果缓存不存在，抓取全市场
    st.info("正在抓取全市场股票数据，请稍等…")
    sh_stocks = fetch_symbol_list("SH")
    sz_stocks = fetch_symbol_list("SZ")
    universe = pd.concat([sh_stocks, sz_stocks], ignore_index=True)
    if universe.empty:
        st.error("获取股票列表失败")
        st.stop()

    records = []
    batch_size = 50
    for start in range(0, len(universe), batch_size):
        batch = universe.iloc[start:start+batch_size]
        for _, row in batch.iterrows():
            region = row["region"]    # iTick返回的交易所字段
            code = row["symbol"]      # iTick返回的股票代码字段
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
        time.sleep(1)  # 延时避免超限
    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)
    return df

# ----------------------------
# 加载数据
# ----------------------------
df = load_data()
if df.empty:
    st.error("个股数据为空")
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
