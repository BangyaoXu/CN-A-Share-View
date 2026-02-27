# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import akshare as ak
from datetime import datetime

st.set_page_config(layout="wide")

st.title("🇨🇳 A股 T+1 主动交易系统（自动数据版）")

# =========================
# 数据获取函数
# =========================

@st.cache_data(ttl=600)
def get_index_data():
    df = ak.stock_zh_index_daily(symbol="sh000001")
    df["MA200"] = df["close"].rolling(200).mean()
    return df

@st.cache_data(ttl=600)
def get_north_money():
    try:
        df = ak.stock_hsgt_hist_em()
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_limit_up():
    df = ak.stock_zt_pool_em(date=datetime.now().strftime("%Y%m%d"))
    return df

@st.cache_data(ttl=600)
def get_sector():
    df = ak.stock_board_industry_name_em()
    return df

# =========================
# 宏观评分
# =========================

index_df = get_index_data()
latest = index_df.iloc[-1]

macro_score = 50
if latest["close"] > latest["MA200"]:
    macro_score += 25

north_df = get_north_money()
if not north_df.empty:
    north_today = north_df.iloc[-1]["当日净流入"]
else:
    north_today = 0

liquidity_score = 50
if north_today > 0:
    liquidity_score += 20

limit_df = get_limit_up()
limit_count = len(limit_df)

sentiment_score = min(limit_count, 100)

total_score = np.mean([macro_score, liquidity_score, sentiment_score])

# =========================
# 仪表盘函数
# =========================

def gauge(title, value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={'axis': {'range': [0,100]}}
    ))
    fig.update_layout(height=250)
    return fig

st.subheader("📊 风险评分")

col1, col2, col3 = st.columns(3)
col1.plotly_chart(gauge("宏观评分", macro_score))
col2.plotly_chart(gauge("流动性评分", liquidity_score))
col3.plotly_chart(gauge("情绪评分", sentiment_score))

st.markdown(f"## 🔥 综合评分：{round(total_score,1)}")

# =========================
# 板块监控
# =========================

st.subheader("🔥 板块涨幅排名")

sector_df = get_sector()
sector_df = sector_df.sort_values(by="涨跌幅", ascending=False).head(10)

st.dataframe(sector_df[["板块名称","涨跌幅"]], use_container_width=True)

strong_sectors = sector_df.head(3)["板块名称"].tolist()

# =========================
# 个股扫描器
# =========================

st.subheader("🔍 个股自动扫描")

stock_list = ak.stock_zh_a_spot_em()
stock_list = stock_list.sort_values(by="涨跌幅", ascending=False).head(200)

candidates = []

for _, row in stock_list.iterrows():
    try:
        code = row["代码"]
        hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20240101")
        if len(hist) < 60:
            continue

        hist["MA60"] = hist["收盘"].rolling(60).mean()

        # 条件1：突破60日新高
        if hist["收盘"].iloc[-1] > hist["收盘"].rolling(60).max().iloc[-2]:
            
            # 条件2：量能放大
            if hist["成交量"].iloc[-1] > 2 * hist["成交量"].rolling(20).mean().iloc[-1]:
                
                # 条件3：不过度乖离
                if (hist["收盘"].iloc[-1] / hist["MA60"].iloc[-1] - 1) < 0.25:
                    
                    candidates.append({
                        "代码": code,
                        "名称": row["名称"],
                        "涨幅": row["涨跌幅"]
                    })
    except:
        continue

candidate_df = pd.DataFrame(candidates)

if len(candidate_df) > 0:
    st.dataframe(candidate_df, use_container_width=True)
else:
    st.info("当前无符合条件个股")

# =========================
# 操作建议
# =========================

st.subheader("🎯 今日操作建议")

if total_score > 70:
    st.success("进攻模式：聚焦强势板块龙头，回踩买入")
elif total_score > 40:
    st.warning("精选模式：控制仓位，快进快出")
else:
    st.error("防守模式：降低仓位，避免追高")

st.caption(f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
