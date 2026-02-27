# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import akshare as ak
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🇨🇳 A股主动交易系统 Ultimate V3.1（板块龙头 + 资金流加权）")

# =========================
# 数据获取函数
# =========================

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
        df = ak.stock_board_industry_name_em()
        if df.empty:
            df = ak.stock_board_concept_name_em()
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_stock_spot():
    try:
        return ak.stock_zh_a_spot_em()
    except:
        return pd.DataFrame()

# =========================
# 风险评分
# =========================
index_df = get_index_data()
macro_score = 50
if not index_df.empty and len(index_df) > 200:
    latest = index_df.iloc[-1]
    if latest["close"] > latest["MA200"]:
        macro_score += 25

north_df = get_north_money()
north_today = 0
if not north_df.empty:
    cols = [c for c in north_df.columns if "净流入" in c]
    if cols:
        north_today = float(north_df.iloc[-1][cols[0]])
liquidity_score = 50 + (20 if north_today > 0 else 0)

limit_df = get_limit_up()
sentiment_score = min(len(limit_df), 100)

total_score = round(np.mean([macro_score, liquidity_score, sentiment_score]), 1)

# =========================
# 仪表盘
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
col1.plotly_chart(gauge("宏观评分", macro_score), use_container_width=True)
col2.plotly_chart(gauge("流动性评分", liquidity_score), use_container_width=True)
col3.plotly_chart(gauge("情绪评分", sentiment_score), use_container_width=True)
st.markdown(f"## 🔥 综合评分：{total_score}")

# =========================
# 板块热度 + 板块龙头
# =========================
st.subheader("🔥 板块热度 & 板块龙头")

sector_df = get_sector()
strong_sectors = []
if not sector_df.empty:
    # 板块热度 = 涨跌幅 + 换手率 + 北向资金流入占比
    sector_df["热度"] = sector_df.get("涨跌幅",0) + sector_df.get("换手率",0)
    # 简单加权资金流
    sector_df["资金流加权"] = 0
    if north_today > 0:
        total_sector_count = len(sector_df)
        sector_df["资金流加权"] = north_today / total_sector_count
        sector_df["热度"] += sector_df["资金流加权"]
    sector_df = sector_df.sort_values(by="热度", ascending=False).head(10)
    strong_sectors = sector_df.head(3)["板块名称"].tolist()
    st.dataframe(sector_df[["板块名称","涨跌幅","换手率","资金流加权","热度"]], use_container_width=True)
else:
    st.info("板块数据获取失败")

# =========================
# 个股扫描 + 板块龙头
# =========================
st.subheader("🔍 板块龙头个股扫描")
stock_list = get_stock_spot()
candidates = []
top_stocks_per_sector = {}

if not stock_list.empty and strong_sectors:
    stock_list = stock_list.sort_values(by="涨跌幅", ascending=False).head(100)
    for _, row in stock_list.iterrows():
        try:
            sector_name = row.get("所属板块","未知")
            if sector_name not in strong_sectors:
                continue
            score = row.get("涨跌幅",0) + row.get("换手率",0)
            candidates.append({
                "代码": row["代码"],
                "名称": row["名称"],
                "板块": sector_name,
                "涨幅": row.get("涨跌幅",0),
                "换手率": row.get("换手率",0),
                "评分": score
            })
            # 每板块只保留Top3
            if sector_name not in top_stocks_per_sector:
                top_stocks_per_sector[sector_name] = []
            top_stocks_per_sector[sector_name].append((score,row["代码"],row["名称"],row.get("涨跌幅",0)))
        except:
            continue

# 取每板块 Top3 个股
final_top_stocks = []
for s, lst in top_stocks_per_sector.items():
    lst.sort(reverse=True)
    for i, item in enumerate(lst[:3]):
        final_top_stocks.append({
            "板块": s,
            "排名": i+1,
            "代码": item[1],
            "名称": item[2],
            "涨幅": item[3]
        })

if final_top_stocks:
    st.dataframe(pd.DataFrame(final_top_stocks), use_container_width=True)
else:
    st.info("当前无板块龙头个股")

# =========================
# 今日操作建议
# =========================
st.subheader("🎯 今日操作建议")
if total_score > 70:
    st.success("进攻模式：聚焦强势板块龙头，回踩买入")
elif total_score > 40:
    st.warning("精选模式：控制仓位，快进快出")
else:
    st.error("防守模式：降低仓位，避免追高")

st.caption(f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =========================
# 板块热力图
# =========================
try:
    if not sector_df.empty:
        fig = px.bar(sector_df, x="板块名称", y="热度", color="热度",
                     text="涨跌幅", title="板块热度排行榜")
        st.plotly_chart(fig, use_container_width=True)
except:
    pass
