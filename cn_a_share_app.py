# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Attempt to import akshare – if missing, show instructions
try:
    import akshare as ak
except ImportError:
    st.error("请先安装 akshare：pip install akshare")
    st.stop()

st.set_page_config(layout="wide")
st.title("🇨🇳 CSI 300 T+1 主动交易系统 (实时数据)")

# ------------------------------------------------------------
# Cached functions to fetch data (TTL = 1 hour for quotes, longer for constituents/sectors)
# ------------------------------------------------------------
@st.cache_data(ttl=3600)  # 1 hour
def get_constituents():
    """获取沪深300成分股列表 (代码, 名称)"""
    try:
        df = ak.index_stock_cons(symbol="000300")
        # 保留所需列，并统一列名
        df = df[["品种代码", "品种名称"]].rename(columns={
            "品种代码": "code",
            "品种名称": "name"
        })
        # 转换为字符串并补零至6位（akshare 有时返回 "000001" 形式，确保一致）
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    except Exception as e:
        st.error(f"获取成分股失败：{e}")
        return pd.DataFrame(columns=["code", "name"])

@st.cache_data(ttl=86400)  # 1 day
def get_sector_mapping():
    """获取全A股行业分类（东方财富版）"""
    try:
        df = ak.stock_industry_clf_em()
        # 列名示例：'代码', '名称', '行业', ...
        df = df[["代码", "行业"]].rename(columns={"代码": "code", "行业": "sector"})
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    except Exception as e:
        st.warning(f"获取行业分类失败：{e}，将使用模拟板块")
        return pd.DataFrame(columns=["code", "sector"])

@st.cache_data(ttl=3600)  # 1 hour
def get_stock_quote(code):
    """获取单只股票的最新日线行情（前复权）"""
    try:
        # 获取最近20个交易日，避免停牌等情况取不到最新
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20000101", adjust="qfq")
        if df.empty:
            return None
        # 取最后一行（最新交易日）
        last = df.iloc[-1]
        # 计算成交额（元） = 成交量(股) * 收盘价
        turnover = last["成交量"] * last["收盘"]
        return {
            "close": last["收盘"],
            "volume": last["成交量"],
            "pct_chg": last["涨跌幅"],
            "turnover": turnover,
            "date": last["日期"]
        }
    except Exception:
        return None

# ------------------------------------------------------------
# Main data acquisition with progress bar
# ------------------------------------------------------------
progress_placeholder = st.empty()
bar_placeholder = st.progress(0.0)

# 1. 获取成分股
progress_placeholder.text("获取沪深300成分股列表...")
constituents = get_constituents()
if constituents.empty:
    st.error("无法获取成分股，请检查网络或稍后重试")
    st.stop()

# 2. 获取行业映射
progress_placeholder.text("获取行业分类...")
sector_map = get_sector_mapping()

# 将成分股与行业合并
merged = constituents.merge(sector_map, on="code", how="left")
merged["sector"] = merged["sector"].fillna("其他")  # 未匹配到的设为“其他”

# 3. 逐只股票获取最新行情
records = []
total = len(merged)

for idx, row in merged.iterrows():
    code = row["code"]
    name = row["name"]
    sector = row["sector"]
    
    quote = get_stock_quote(code)
    if quote:
        records.append({
            "代码": code,
            "名称": name,
            "板块": sector,
            "涨跌幅": quote["pct_chg"],
            "成交量": quote["turnover"]          # 成交额（元）
        })
    else:
        # 如果取不到行情，用空值占位，后续会被过滤掉
        records.append({
            "代码": code,
            "名称": name,
            "板块": sector,
            "涨跌幅": np.nan,
            "成交量": np.nan
        })
    
    # 更新进度
    progress_placeholder.text(f"抓取 CSI300 行情: {idx+1}/{total}")
    bar_placeholder.progress((idx+1)/total)
    time.sleep(0.1)  # 控制请求频率，避免被封

progress_placeholder.text("CSI300 数据抓取完成！")
bar_placeholder.progress(1.0)

# 4. 构建DataFrame，并删除无行情的股票
df = pd.DataFrame(records).dropna(subset=["涨跌幅", "成交量"])
if df.empty:
    st.error("未能获取任何有效行情数据，请稍后重试")
    st.stop()

# ------------------------------------------------------------
# 板块热度排行榜 (使用成交额，单位：十亿元，以使数值与涨跌幅量级相近)
# ------------------------------------------------------------
sector_score = df.groupby("板块").agg({
    "涨跌幅": "mean",
    "成交量": "sum"
}).reset_index()
# 热度 = 平均涨跌幅 + 总成交额 / 1e9  （将十亿元转换为“点”）
sector_score["热度"] = sector_score["涨跌幅"] + sector_score["成交量"] / 1e9
sector_score = sector_score.sort_values("热度", ascending=False)
top_sectors = sector_score.head(10)

st.subheader("🔥 板块热度排行榜")
st.dataframe(top_sectors, use_container_width=True)

# ------------------------------------------------------------
# 板块龙头个股
# ------------------------------------------------------------
df["评分"] = df["涨跌幅"] + df["成交量"] / 1e9   # 与板块热度一致
top_stocks = df.sort_values("评分", ascending=False).groupby("板块").head(3)

st.subheader("🔍 板块龙头个股")
st.dataframe(top_stocks[["板块", "代码", "名称", "涨跌幅", "成交量"]], use_container_width=True)

# ------------------------------------------------------------
# 综合评分 (宏观/流动性/情绪 – 模拟指标，可根据实际情况调整)
# ------------------------------------------------------------
macro_score = 50
liquidity_score = 50
sentiment_score = min(len(top_stocks) * 10, 100)   # 简单示例：每只龙头股贡献10分
total_score = np.mean([macro_score, liquidity_score, sentiment_score])

def gauge(title, value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={'axis': {'range': [0, 100]}}
    ))
    fig.update_layout(height=250)
    return fig

st.subheader("📊 综合评分")
col1, col2, col3 = st.columns(3)
col1.plotly_chart(gauge("宏观评分", macro_score))
col2.plotly_chart(gauge("流动性评分", liquidity_score))
col3.plotly_chart(gauge("情绪评分", sentiment_score))
st.markdown(f"## 🔥 综合评分: {round(total_score, 1)}")

# ------------------------------------------------------------
# 今日操作建议
# ------------------------------------------------------------
st.subheader("🎯 今日操作建议")
if total_score > 70:
    st.success("进攻模式：聚焦强势板块龙头，回踩买入")
elif total_score > 40:
    st.warning("精选模式：控制仓位，快进快出")
else:
    st.error("防守模式：降低仓位，避免追高")

st.caption(f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------------------------------------
# 板块热力图
# ------------------------------------------------------------
fig = px.bar(top_sectors, x="板块", y="热度", color="热度",
             text="涨跌幅", title="板块热度排行榜")
st.plotly_chart(fig, use_container_width=True)
