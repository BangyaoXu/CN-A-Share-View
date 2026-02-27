# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import akshare as ak
from datetime import datetime

st.set_page_config(layout="wide")

st.title("🇨🇳 A股 T+1 主动交易系统（自动数据稳定版）")

# =====================================================
# 数据获取函数（全部加防炸保护）
# =====================================================

@st.cache_data(ttl=600)
def get_index_data():
    try:
        df = ak.stock_zh_index_daily(symbol="sh000001")
        df["MA200"] = df["close"].rolling(200).mean()
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_north_money():
    try:
        return ak.stock_hsgt_hist_em()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_limit_up():
    try:
        return ak.stock_zt_pool_em(date=datetime.now().strftime("%Y%m%d"))
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_sector():
    try:
        return ak.stock_board_industry_name_em()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_stock_spot():
    try:
        return ak.stock_zh_a_spot_em()
    except:
        return pd.DataFrame()

def get_today_north_flow():
    df = get_north_money()
    if df.empty:
        return 0

    possible_cols = [col for col in df.columns if "净流入" in col]
    if not possible_cols:
        return 0

    try:
        return float(df.iloc[-1][possible_cols[0]])
    except:
        return 0


# =====================================================
# 宏观评分
# =====================================================

index_df = get_index_data()

macro_score = 50
if not index_df.empty and len(index_df) > 200:
    latest = index_df.iloc[-1]
    if latest["close"] > latest["MA200"]:
        macro_score += 25

north_today = get_today_north_flow()

liquidity_score = 50
if north_today > 0:
    liquidity_score += 20

limit_df = get_limit_up()
limit_count = len(limit_df)
sentiment_score = min(limit_count, 100)

total_score = round(np.mean([macro_score, liquidity_score, sentiment_score]), 1)


# =====================================================
# 仪表盘
# =====================================================

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
col1.plotly_chart(gauge("宏观评分", macro_score), use_container_width=True)
col2.plotly_chart(gauge("流动性评分", liquidity_score), use_container_width=True)
col3.plotly_chart(gauge("情绪评分", sentiment_score), use_container_width=True)

st.markdown(f"## 🔥 综合评分：{total_score}")


# =====================================================
# 板块监控
# =====================================================

st.subheader("🔥 板块涨幅排名")

sector_df = get_sector()

if not sector_df.empty and "涨跌幅" in sector_df.columns:
    sector_df = sector_df.sort_values(by="涨跌幅", ascending=False).head(10)
    st.dataframe(sector_df[["板块名称","涨跌幅"]], use_container_width=True)
    strong_sectors = sector_df.head(3)["板块名称"].tolist()
else:
    strong_sectors = []
    st.info("板块数据获取失败")


# =====================================================
# 个股扫描器（限制扫描数量避免超时）
# =====================================================

st.subheader("🔍 个股自动扫描")

stock_list = get_stock_spot()

candidates = []

if not stock_list.empty and "涨跌幅" in stock_list.columns:
    
    # 只扫描前100只，防止云端超时
    stock_list = stock_list.sort_values(by="涨跌幅", ascending=False).head(100)

    progress = st.progress(0)
    total = len(stock_list)

    for i, (_, row) in enumerate(stock_list.iterrows()):
        try:
            code = row["代码"]

            hist = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date="20240101"
            )

            if len(hist) < 60:
                continue

            hist["MA60"] = hist["收盘"].rolling(60).mean()

            # 条件1：突破60日新高
            cond1 = hist["收盘"].iloc[-1] > hist["收盘"].rolling(60).max().iloc[-2]

            # 条件2：量能放大
            cond2 = hist["成交量"].iloc[-1] > 2 * hist["成交量"].rolling(20).mean().iloc[-1]

            # 条件3：不过度乖离
            cond3 = (hist["收盘"].iloc[-1] / hist["MA60"].iloc[-1] - 1) < 0.25

            if cond1 and cond2 and cond3:
                candidates.append({
                    "代码": code,
                    "名称": row["名称"],
                    "涨幅": row["涨跌幅"]
                })

        except:
            continue

        progress.progress((i + 1) / total)

candidate_df = pd.DataFrame(candidates)

if not candidate_df.empty:
    st.dataframe(candidate_df, use_container_width=True)
else:
    st.info("当前无符合条件个股")


# =====================================================
# 操作建议
# =====================================================

st.subheader("🎯 今日操作建议")

if total_score > 70:
    st.success("进攻模式：聚焦强势板块龙头，回踩买入")
elif total_score > 40:
    st.warning("精选模式：控制仓位，快进快出")
else:
    st.error("防守模式：降低仓位，避免追高")

st.caption(f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
