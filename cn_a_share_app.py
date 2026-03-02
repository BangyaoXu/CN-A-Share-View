# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import time

st.set_page_config(layout="wide", page_title="CSI 300 真实数据仪表盘", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; font-weight: 800; text-align: center; }
    .sub-header { font-size: 1rem; color: #6B7280; text-align: center; }
    .section-header { font-size: 1.5rem; font-weight: 600; color: #1E3A8A; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; margin: 2rem 0 1rem 0; }
    .insight-box { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; border-left: 4px solid #4F46E5; margin: 1rem 0; color: #000000; }
    .strategy-box { background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border: 1px solid #e5e7eb; margin: 1rem 0; color: #000000; }
    .metric-label { color: #4B5563; font-size: 0.9rem; }
    .metric-value { color: #111827; font-size: 1.2rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load constituent list from CSV
# ------------------------------------------------------------
@st.cache_data(ttl=86400)
def load_constituents():
    df = pd.read_csv('csi300_full.csv')
    df['code'] = df['code'].astype(str).str.zfill(6)
    return df

# ------------------------------------------------------------
# Yahoo Finance helper
# ------------------------------------------------------------
def code_to_yf(code):
    code = str(code).zfill(6)
    # Shanghai stocks usually start with 5 or 6, Shenzhen with 0,2,3
    return f"{code}.SS" if code.startswith(('5','6')) else f"{code}.SZ"

# ------------------------------------------------------------
# Fetch real-time stock data (cached 15 min)
# ------------------------------------------------------------
@st.cache_data(ttl=900)
def fetch_realtime_stocks(ticker_list):
    stocks = []
    prog = st.progress(0)
    status = st.empty()
    total = len(ticker_list)
    for i, (code, name, sector) in enumerate(ticker_list):
        status.text(f"获取 {i+1}/{total}: {name}")
        yf_ticker = code_to_yf(code)
        try:
            stock = yf.Ticker(yf_ticker)
            hist = stock.history(period="5d")
            if not hist.empty:
                last = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else last
                pct = ((last['Close'] - prev['Close']) / prev['Close']) * 100
                stocks.append({
                    '代码': code,
                    '名称': name,
                    '板块': sector,
                    '最新价': round(last['Close'], 2),
                    '涨跌幅': round(pct, 2),
                    '成交量': last['Volume'],
                    '成交额(亿)': round(last['Volume'] * last['Close'] / 1e8, 2),
                    '最高': round(last['High'], 2),
                    '最低': round(last['Low'], 2),
                    '开盘': round(last['Open'], 2),
                })
        except Exception:
            # skip failed stocks
            pass
        prog.progress((i+1)/total)
        time.sleep(0.1)
    status.empty()
    prog.empty()
    return pd.DataFrame(stocks)

# ------------------------------------------------------------
# Index historical data with EMAs
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_index_hist(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df.empty:
            return None
        df = df[['Close']].copy()
        df.columns = ['close']
        for span in [20, 60, 120, 250]:   # EMA 250 instead of 150
            df[f'EMA{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        return df
    except:
        return None

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    st.markdown('<p class="main-header">📊 CSI 300 真实数据仪表盘</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">实时行情 + 技术分析 + 板块轮动</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/investment-portfolio.png", width=100)
        st.title("控制面板")
        if st.button("🔄 刷新所有数据", type="primary"):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.markdown("### 数据源")
        st.info("📈 股价: Yahoo Finance")
        st.info("📊 成分股: 本地 CSV")
        st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.warning("新闻与情绪数据暂不可用（无可靠免费API）")

    constituents = load_constituents()
    ticker_list = list(zip(constituents['code'], constituents['name'], constituents['sector']))

    with st.spinner("获取实时行情..."):
        df = fetch_realtime_stocks(ticker_list)

    if df.empty:
        st.error("未能获取任何股票数据，请检查网络")
        st.stop()

    # --- Index charts with EMAs ---
    st.markdown("### 📈 主要指数技术分析")
    indices = {
        '000300.SS': '沪深300',
        '000001.SS': '上证指数',
        '399001.SZ': '深证成指'
    }
    tabs = st.tabs(list(indices.values()))
    for i, (ticker, name) in enumerate(indices.items()):
        with tabs[i]:
            hist_df = get_index_hist(ticker, period="6mo")
            if hist_df is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['close'], mode='lines', name='收盘价'))
                for span in [20,60,120,250]:
                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df[f'EMA{span}'], mode='lines', name=f'EMA{span}'))
                fig.update_layout(height=500, title=f"{name} 日线图 (EMA 20/60/120/250)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"{name} 数据获取失败")

    # --- Key Metrics Row ---
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    avg_ret = df['涨跌幅'].mean()
    pos_ratio = (df['涨跌幅'] > 0).mean() * 100
    total_vol = df['成交额(亿)'].sum()
    vola = df['涨跌幅'].std()
    with col1:
        st.metric("平均涨跌幅", f"{avg_ret:.2f}%")
    with col2:
        st.metric("上涨比例", f"{pos_ratio:.1f}%", delta=f"{pos_ratio-50:.1f}%")
    with col3:
        st.metric("总成交额(亿)", f"{total_vol:.0f}")
    with col4:
        st.metric("波动率", f"{vola:.2f}%")
    with col5:
        # Simple sentiment proxy based on breadth
        breadth_sentiment = 50 + (pos_ratio - 50)
        st.metric("市场宽度", f"{breadth_sentiment:.0f}")

    # --- Market Insight ---
    best_sector = df.groupby('板块')['涨跌幅'].mean().idxmax()
    best_ret = df.groupby('板块')['涨跌幅'].mean().max()
    st.markdown(f"""
    <div class="insight-box">
        <strong>📊 市场实时洞察</strong><br>
        强势板块: {best_sector} (+{best_ret:.2f}%) |
        波动风险: {'高' if vola>2 else '中' if vola>1 else '低'} |
        上涨家数: {int(pos_ratio*len(df)/100)} / {len(df)}
    </div>
    """, unsafe_allow_html=True)

    # --- Sector Analysis ---
    st.markdown('<div class="section-header">🏭 板块深度分析</div>', unsafe_allow_html=True)
    sector_stats = df.groupby('板块').agg(
        平均涨跌幅=('涨跌幅','mean'),
        涨跌中位数=('涨跌幅','median'),
        波动率=('涨跌幅','std'),
        成分股数=('代码','count'),
        总成交额=('成交额(亿)','sum'),
        平均成交额=('成交额(亿)','mean')
    ).reset_index().round(2)
    sector_stats = sector_stats.sort_values('平均涨跌幅', ascending=False)

    # Bubble chart
    fig = px.scatter(sector_stats, x='平均涨跌幅', y='总成交额', size='成分股数',
                     color='平均涨跌幅', text='板块', title='板块轮动气泡图',
                     color_continuous_scale='RdYlGn', size_max=50)
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

    # Bar chart
    fig = px.bar(sector_stats.head(10), x='板块', y='平均涨跌幅', color='平均涨跌幅',
                 text='平均涨跌幅', title='板块涨跌幅前十',
                 color_continuous_scale=['#EF4444','#FCD34D','#10B981'])
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(sector_stats, use_container_width=True, hide_index=True)

    # --- Sector Stock Selector ---
    st.markdown('<div class="section-header">🔍 板块内选股</div>', unsafe_allow_html=True)
    selected_sector = st.selectbox("选择板块", ['全部'] + sorted(df['板块'].unique()))
    if selected_sector != '全部':
        sector_df = df[df['板块'] == selected_sector].copy()
    else:
        sector_df = df.copy()

    # Multi‑factor scoring (momentum + volume)
    sector_df['动量分'] = (sector_df['涨跌幅'] - sector_df['涨跌幅'].mean()) / sector_df['涨跌幅'].std()
    sector_df['成交额分'] = (sector_df['成交额(亿)'] - sector_df['成交额(亿)'].mean()) / sector_df['成交额(亿)'].std()
    sector_df['综合分'] = (sector_df['动量分']*0.6 + sector_df['成交额分']*0.4).round(2)

    top_stocks = sector_df.nlargest(15, '综合分')[['代码','名称','板块','最新价','涨跌幅','成交额(亿)','综合分']]
    top_stocks['涨跌幅'] = top_stocks['涨跌幅'].apply(lambda x: f"{x:.2f}%")
    st.dataframe(top_stocks, use_container_width=True, hide_index=True)

    # --- Top Movers ---
    st.markdown('<div class="section-header">📈 全市场龙虎榜</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("涨幅前十")
        gain = df.nlargest(10, '涨跌幅')[['代码','名称','板块','最新价','涨跌幅','成交额(亿)']]
        gain['涨跌幅'] = gain['涨跌幅'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(gain, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("跌幅前十")
        lose = df.nsmallest(10, '涨跌幅')[['代码','名称','板块','最新价','涨跌幅','成交额(亿)']]
        lose['涨跌幅'] = lose['涨跌幅'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(lose, use_container_width=True, hide_index=True)

    # --- Strategy Recommendation ---
    st.markdown('<div class="section-header">📋 实时策略建议</div>', unsafe_allow_html=True)
    if avg_ret > 0.5 and pos_ratio > 60:
        regime = "牛市"
        color = "#10B981"
        pos = "70-80%"
    elif avg_ret < -0.5 and pos_ratio < 40:
        regime = "熊市"
        color = "#EF4444"
        pos = "20-30%"
    elif vola > 2:
        regime = "高波动市"
        color = "#F59E0B"
        pos = "40-50%"
    else:
        regime = "震荡市"
        color = "#4F46E5"
        pos = "50-60%"

    top_sectors = sector_stats.head(3)['板块'].tolist()
    bottom_sectors = sector_stats.tail(3)['板块'].tolist()
    st.markdown(f"""
    <div class="strategy-box">
        <h3>当前市场状态: <span style="color:{color};">{regime}</span></h3>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem;">
            <div><span class="metric-label">建议仓位</span><div class="metric-value">{pos}</div></div>
            <div><span class="metric-label">风险水平</span><div class="metric-value">{'高' if vola>2 else '中' if vola>1 else '低'}</div></div>
            <div><span class="metric-label">操作方向</span><div class="metric-value">{'积极' if avg_ret>0 else '防御'}</div></div>
        </div>
        <div style="margin-top:1.5rem; border-top:1px solid #ddd; padding-top:1rem;">
            <p><strong>重点关注板块:</strong> {', '.join(top_sectors)}</p>
            <p><strong>建议规避板块:</strong> {', '.join(bottom_sectors)}</p>
            <p><strong>选股参考:</strong> 板块内综合分 > 0 | 成交额 > 板块平均</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:#666; font-size:0.8rem;">
        数据来源: Yahoo Finance | 成分股列表: csi300_full.csv | 更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
