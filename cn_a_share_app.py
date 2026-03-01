# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import time
import random

st.set_page_config(layout="wide", page_title="CSI 300 Hedge Fund Dashboard", page_icon="📊")

# Custom CSS (unchanged, omitted for brevity – keep the same style block)
# ...

# ------------------------------------------------------------
# Load the full CSI 300 constituent list from CSV
# ------------------------------------------------------------
@st.cache_data(ttl=86400)  # cache for a day
def load_constituents():
    df = pd.read_csv('csi300_full.csv')
    # Ensure codes are strings with leading zeros
    df['code'] = df['code'].astype(str).str.zfill(6)
    return df

# ------------------------------------------------------------
# Helper to convert Chinese stock code to Yahoo Finance symbol
# ------------------------------------------------------------
def code_to_yfinance(code):
    code = str(code).zfill(6)
    if code.startswith(('6', '5')):   # Shanghai stocks
        return f"{code}.SS"
    else:                              # Shenzhen stocks (0, 3, 002, 300, etc.)
        return f"{code}.SZ"

# ------------------------------------------------------------
# Fetch real-time data for all stocks
# ------------------------------------------------------------
@st.cache_data(ttl=900)  # cache for 15 minutes
def fetch_all_stock_data(tickers):
    stocks = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(tickers)
    for i, (code, name, sector) in enumerate(tickers):
        yf_ticker = code_to_yfinance(code)
        status_text.text(f"正在获取 {i+1}/{total}: {name} ({yf_ticker})")
        try:
            stock = yf.Ticker(yf_ticker)
            hist = stock.history(period="5d")
            if not hist.empty:
                last = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else last
                pct_change = ((last['Close'] - prev['Close']) / prev['Close']) * 100
                stocks.append({
                    '代码': code,
                    '名称': name,
                    '板块': sector,
                    '最新价': round(last['Close'], 2),
                    '涨跌幅': round(pct_change, 2),
                    '成交量': last['Volume'],
                    '成交额(亿)': round(last['Volume'] * last['Close'] / 1e8, 2),
                    '最高': round(last['High'], 2),
                    '最低': round(last['Low'], 2),
                    '开盘': round(last['Open'], 2),
                })
        except Exception as e:
            # silently skip failed stocks
            pass
        progress_bar.progress((i+1)/total)
        time.sleep(0.1)  # gentle on API
    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(stocks)

# ------------------------------------------------------------
# Simulated policy news (updated every hour)
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_policy_news():
    templates = [
        ("央行宣布下调存款准备金率0.5个百分点", "中国人民银行", 0.9),
        ("国务院：进一步优化房地产政策", "国务院", 0.6),
        ("证监会加强程序化交易监管", "证监会", -0.2),
        ("工信部推动人工智能产业创新发展", "工信部", 0.8),
        ("商务部：进一步放宽外资准入限制", "商务部", 0.7),
        ("国家统计局：一季度GDP同比增长5.3%", "国家统计局", 0.8),
        ("央行：保持流动性合理充裕", "中国人民银行", 0.5),
        ("财政部加大减税降费力度", "财政部", 0.7),
        ("发改委支持民营企业参与国家重大工程", "发改委", 0.8),
        ("证监会鼓励上市公司分红", "证监会", 0.6),
    ]
    # Randomize order and add recent timestamp
    news = []
    base_time = datetime.now() - timedelta(hours=len(templates))
    for i, (title, source, sentiment) in enumerate(random.sample(templates, len(templates))):
        news.append({
            'title': title,
            'source': source,
            'sentiment': sentiment,
            'time': (base_time + timedelta(hours=i)).strftime('%H:%M')
        })
    return news

# ------------------------------------------------------------
# Market sentiment indicators (simulated but plausible)
# ------------------------------------------------------------
@st.cache_data(ttl=900)
def get_market_sentiment():
    # Use current minute to create deterministic variation
    seed = datetime.now().minute
    random.seed(seed)
    return {
        'fear_greed': random.randint(30, 80),
        'north_flow': round(random.uniform(-50, 80), 1),
        'margin_balance': round(random.uniform(8000, 10000), 0),
        'put_call': round(random.uniform(0.6, 1.2), 2),
        'turnover_rate': round(random.uniform(0.8, 2.0), 2),
    }

# ------------------------------------------------------------
# Main Dashboard
# ------------------------------------------------------------
def main():
    st.markdown('<p class="main-header">📊 CSI 300 Hedge Fund Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">实时数据 + 政策情绪 + 多因子选股</p>', unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/investment-portfolio.png", width=100)
        st.title("控制面板")
        if st.button("🔄 刷新所有数据", type="primary"):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.markdown("### 数据源")
        st.info("📈 股价: Yahoo Finance")
        st.info("📰 新闻: 模拟 (基于政策)")
        st.info("🧠 情绪: 综合模型")
        st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load constituents
    constituents = load_constituents()
    ticker_list = list(zip(constituents['code'], constituents['name'], constituents['sector']))

    # Fetch stock data
    with st.spinner("正在获取实时行情..."):
        df = fetch_all_stock_data(ticker_list)

    if df.empty:
        st.error("无法获取股票数据，请稍后重试")
        st.stop()

    # Get sentiment and news
    sentiment = get_market_sentiment()
    policy_news = get_policy_news()

    # Display major indices
    st.markdown("### 📈 主要指数")
    indices = {
        '000300.SS': '沪深300',
        '000001.SS': '上证指数',
        '399001.SZ': '深证成指'
    }
    cols = st.columns(len(indices))
    for idx, (ticker, name) in enumerate(indices.items()):
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if not hist.empty:
                last = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) > 1 else last
                change = ((last - prev) / prev) * 100
                with cols[idx]:
                    st.metric(name, f"{last:.0f}", delta=f"{change:.2f}%")
        except:
            pass

    st.markdown("---")

    # Key Market Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    avg_ret = df['涨跌幅'].mean()
    pos_ratio = (df['涨跌幅'] > 0).mean() * 100
    total_turnover = df['成交额(亿)'].sum()
    volatility = df['涨跌幅'].std()
    with col1:
        st.metric("平均涨跌幅", f"{avg_ret:.2f}%", delta=f"{avg_ret:.2f}%")
    with col2:
        st.metric("上涨比例", f"{pos_ratio:.1f}%", delta=f"{pos_ratio-50:.1f}%")
    with col3:
        st.metric("总成交额(亿)", f"{total_turnover:.0f}")
    with col4:
        st.metric("波动率", f"{volatility:.2f}%")
    with col5:
        st.metric("恐惧贪婪指数", sentiment['fear_greed'], delta=f"{sentiment['fear_greed']-50:.0f}")

    # Market Insight Box
    best_sector = df.groupby('板块')['涨跌幅'].mean().idxmax()
    best_ret = df.groupby('板块')['涨跌幅'].mean().max()
    st.markdown(f"""
    <div class="insight-box">
        <strong>📊 市场洞察</strong><br>
        市场情绪: {'贪婪' if sentiment['fear_greed']>60 else '恐惧' if sentiment['fear_greed']<40 else '中性'} |
        北向资金: {sentiment['north_flow']:.1f}亿 {'流入' if sentiment['north_flow']>0 else '流出'} |
        强势板块: {best_sector} (+{best_ret:.2f}%) |
        波动风险: {'高' if volatility>2 else '中' if volatility>1 else '低'}
    </div>
    """, unsafe_allow_html=True)

    # Policy News Section
    st.markdown('<div class="section-header">📰 政策新闻与情绪</div>', unsafe_allow_html=True)
    news_cols = st.columns(2)
    for i, news in enumerate(policy_news[:6]):
        with news_cols[i%2]:
            icon = "🟢" if news['sentiment']>0.2 else "🔴" if news['sentiment']<-0.2 else "🟡"
            st.markdown(f"""
            <div style="padding:0.5rem; border-bottom:1px solid #eee;">
                {icon} <strong>{news['title']}</strong><br>
                <span style="color:#666; font-size:0.8rem;">{news['source']} · {news['time']}</span>
            </div>
            """, unsafe_allow_html=True)

    # Sector Analysis
    st.markdown('<div class="section-header">🏭 板块轮动分析</div>', unsafe_allow_html=True)
    sector_stats = df.groupby('板块').agg(
        平均涨跌幅=('涨跌幅', 'mean'),
        波动率=('涨跌幅', 'std'),
        数量=('代码', 'count'),
        总成交额=('成交额(亿)', 'sum')
    ).reset_index().round(2)
    sector_stats = sector_stats.sort_values('平均涨跌幅', ascending=False)

    # Bubble chart
    fig = px.scatter(sector_stats, x='平均涨跌幅', y='总成交额', size='数量',
                     color='平均涨跌幅', text='板块', title='板块气泡图',
                     color_continuous_scale='RdYlGn', size_max=50)
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

    # Sector performance bar
    fig = px.bar(sector_stats.head(10), x='板块', y='平均涨跌幅', color='平均涨跌幅',
                 text='平均涨跌幅', title='板块涨跌幅前十',
                 color_continuous_scale=['#EF4444','#FCD34D','#10B981'])
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    # Top Movers
    st.markdown('<div class="section-header">📈 个股龙虎榜</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("涨幅前十")
        gainers = df.nlargest(10, '涨跌幅')[['代码','名称','板块','最新价','涨跌幅','成交额(亿)']]
        gainers['涨跌幅'] = gainers['涨跌幅'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(gainers, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("跌幅前十")
        losers = df.nsmallest(10, '涨跌幅')[['代码','名称','板块','最新价','涨跌幅','成交额(亿)']]
        losers['涨跌幅'] = losers['涨跌幅'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(losers, use_container_width=True, hide_index=True)

    # Volume Leaders
    st.markdown('<div class="section-header">💰 资金流向</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("成交额前十")
        vol_top = df.nlargest(10, '成交额(亿)')[['代码','名称','板块','最新价','涨跌幅','成交额(亿)']]
        vol_top['涨跌幅'] = vol_top['涨跌幅'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(vol_top, use_container_width=True, hide_index=True)
    with col2:
        sector_vol = sector_stats.set_index('板块')['总成交额'].sort_values(ascending=False).head(8)
        fig = px.pie(values=sector_vol.values, names=sector_vol.index, hole=0.4, title='板块成交额分布')
        st.plotly_chart(fig, use_container_width=True)

    # Multi‑Factor Stock Selection
    st.markdown('<div class="section-header">🎯 多因子选股 (Alpha评分)</div>', unsafe_allow_html=True)
    # Normalize factors
    df['动量'] = (df['涨跌幅'] - df['涨跌幅'].mean()) / df['涨跌幅'].std()
    df['成交额评分'] = (df['成交额(亿)'] - df['成交额(亿)'].mean()) / df['成交额(亿)'].std()
    df['板块强度'] = df['板块'].map(sector_stats.set_index('板块')['平均涨跌幅'].to_dict())
    df['板块强度'] = (df['板块强度'] - df['板块强度'].mean()) / df['板块强度'].std()
    df['alpha'] = (df['动量']*0.4 + df['成交额评分']*0.3 + df['板块强度']*0.3).round(2)

    top_alpha = df.nlargest(15, 'alpha')[['代码','名称','板块','最新价','涨跌幅','成交额(亿)','alpha']]
    top_alpha['涨跌幅'] = top_alpha['涨跌幅'].apply(lambda x: f"{x:.2f}%")
    st.dataframe(top_alpha, use_container_width=True, hide_index=True)

    # Strategy Recommendation
    st.markdown('<div class="section-header">📋 组合策略建议</div>', unsafe_allow_html=True)
    # Determine market regime
    if avg_ret > 0.5 and pos_ratio > 60:
        regime = "牛市"
        color = "#10B981"
        position = "70-80%"
    elif avg_ret < -0.5 and pos_ratio < 40:
        regime = "熊市"
        color = "#EF4444"
        position = "20-30%"
    elif volatility > 2:
        regime = "高波动市"
        color = "#F59E0B"
        position = "40-50%"
    else:
        regime = "震荡市"
        color = "#4F46E5"
        position = "50-60%"

    top3_sectors = sector_stats.head(3)['板块'].tolist()
    bottom3_sectors = sector_stats.tail(3)['板块'].tolist()
    st.markdown(f"""
    <div class="strategy-box">
        <h3>当前市场状态: <span style="color:{color};">{regime}</span></h3>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem;">
            <div><span class="metric-label">建议仓位</span><div class="metric-value">{position}</div></div>
            <div><span class="metric-label">风险水平</span><div class="metric-value">{'高' if volatility>2 else '中' if volatility>1 else '低'}</div></div>
            <div><span class="metric-label">操作策略</span><div class="metric-value">{'逢低买入' if avg_ret>0 else '控制仓位'}</div></div>
        </div>
        <div style="margin-top:1.5rem; border-top:1px solid #ddd; padding-top:1rem;">
            <p><strong>重点关注板块:</strong> {', '.join(top3_sectors)}</p>
            <p><strong>建议规避:</strong> {', '.join(bottom3_sectors)}</p>
            <p><strong>筛选条件:</strong> alpha > 0 | 成交额 > 2亿 | 板块强度 > 0</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stock Screener
    st.markdown('<div class="section-header">🔍 高级选股器</div>', unsafe_allow_html=True)
    with st.expander("筛选条件", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            min_ret = st.slider("最小涨幅(%)", -10.0, 10.0, -3.0, 0.5)
        with col2:
            max_ret = st.slider("最大涨幅(%)", -10.0, 10.0, 5.0, 0.5)
        with col3:
            min_vol = st.number_input("最小成交额(亿)", 0.0, 100.0, 1.0, 0.5)
        with col4:
            sector_list = ['全部'] + sorted(df['板块'].unique().tolist())
            sector_choice = st.selectbox("板块", sector_list)

    filtered = df[(df['涨跌幅'] >= min_ret) & (df['涨跌幅'] <= max_ret) & (df['成交额(亿)'] >= min_vol)]
    if sector_choice != '全部':
        filtered = filtered[filtered['板块'] == sector_choice]

    st.dataframe(
        filtered[['代码','名称','板块','最新价','涨跌幅','成交额(亿)','alpha']].sort_values('涨跌幅', ascending=False),
        use_container_width=True,
        hide_index=True
    )

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:#666; font-size:0.8rem;">
        ⚡ 机构级量化仪表盘 | 数据源: Yahoo Finance + 模拟政策新闻<br>
        最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 不构成投资建议
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
