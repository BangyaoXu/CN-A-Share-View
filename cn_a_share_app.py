# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import time

st.set_page_config(layout="wide", page_title="CSI 800 + CSI 1000 Dashboard", page_icon="📊")

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
    df = pd.read_csv('universe.csv')
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
    
    # Calculate date ranges for precise period control
    end_date = datetime.now()
    start_date_1m = (end_date - timedelta(days=35)).strftime('%Y-%m-%d')  # 35 days to ensure 20+ trading days
    
    for i, (code, name, sector) in enumerate(ticker_list):
        status.text(f"获取 {i+1}/{total}: {name}")
        yf_ticker = code_to_yf(code)
        try:
            # Request more data to ensure we have enough for 1m calculation
            stock = yf.Ticker(yf_ticker)
            
            # Get 2 months of data to be safe
            hist = stock.history(start=start_date_1m, end=end_date.strftime('%Y-%m-%d'))
            
            if not hist.empty and len(hist) >= 5:
                last = hist.iloc[-1]
                
                # Calculate returns for different periods
                # 1d return
                if len(hist) >= 2:
                    prev_day = hist.iloc[-2]
                    ret_1d = ((last['Close'] - prev_day['Close']) / prev_day['Close']) * 100
                else:
                    ret_1d = 0
                
                # 1w return (5 trading days)
                if len(hist) >= 5:
                    prev_week = hist.iloc[-5]
                    ret_1w = ((last['Close'] - prev_week['Close']) / prev_week['Close']) * 100
                else:
                    ret_1w = ret_1d
                
                # 1m return (20 trading days)
                if len(hist) >= 20:
                    prev_month = hist.iloc[-20]
                    ret_1m = ((last['Close'] - prev_month['Close']) / prev_month['Close']) * 100
                elif len(hist) >= 10:
                    # If we have at least 10 days, use the oldest available as approximation
                    prev_month = hist.iloc[0]
                    ret_1m = ((last['Close'] - prev_month['Close']) / prev_month['Close']) * 100
                else:
                    # Fallback to 1w return if not enough data
                    ret_1m = ret_1w
                
                stocks.append({
                    '代码': code,
                    '名称': name,
                    '板块': sector,
                    '最新价': round(last['Close'], 2),
                    '涨跌幅_1d': round(ret_1d, 2),
                    '涨跌幅_1w': round(ret_1w, 2),
                    '涨跌幅_1m': round(ret_1m, 2),
                    '成交量': last['Volume'],
                    '成交额(亿)': round(last['Volume'] * last['Close'] / 1e8, 2),
                    '最高': round(last['High'], 2),
                    '最低': round(last['Low'], 2),
                    '开盘': round(last['Open'], 2),
                    '数据点数': len(hist)  # Debug info
                })
        except Exception as e:
            # skip failed stocks
            pass
        prog.progress((i+1)/total)
        time.sleep(0.1)
    status.empty()
    prog.empty()
    
    # Debug info - show data quality
    if not stocks:
        return pd.DataFrame(stocks)
    
    df = pd.DataFrame(stocks)
    
    # Show warning if many stocks have insufficient data
    insufficient_data = df[df['涨跌幅_1m'] == df['涨跌幅_1w']].shape[0]
    if insufficient_data > len(df) * 0.3:  # More than 30% have 1m = 1w
        st.warning(f"⚠️ {insufficient_data} 只股票({insufficient_data/len(df)*100:.1f}%)的1月数据不完整，显示的是1周数据。实际交易天数可能不足20天。")
    
    return df

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
        for span in [20, 60, 120, 250]:
            df[f'EMA{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        return df
    except:
        return None

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    st.markdown('<p class="main-header">📊 CSI 800 + CSI 1000 真实数据仪表盘</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">实时行情 + 技术分析 + 多周期板块轮动</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/investment-portfolio.png", width=100)
        st.title("控制面板")
        if st.button("🔄 刷新所有数据", type="primary"):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.markdown("### 数据源")
        st.info("📈 股价: Yahoo Finance")
        st.info("📊 成分股: universe.csv")
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
    avg_ret_1d = df['涨跌幅_1d'].mean()
    avg_ret_1w = df['涨跌幅_1w'].mean()
    avg_ret_1m = df['涨跌幅_1m'].mean()
    pos_ratio = (df['涨跌幅_1d'] > 0).mean() * 100
    total_vol = df['成交额(亿)'].sum()
    vola = df['涨跌幅_1d'].std()
    
    with col1:
        st.metric("平均涨幅(1日)", f"{avg_ret_1d:.2f}%", delta=f"{avg_ret_1d:.2f}%")
    with col2:
        st.metric("平均涨幅(1周)", f"{avg_ret_1w:.2f}%", delta=f"{avg_ret_1w:.2f}%")
    with col3:
        st.metric("平均涨幅(1月)", f"{avg_ret_1m:.2f}%", delta=f"{avg_ret_1m:.2f}%")
    with col4:
        st.metric("上涨比例", f"{pos_ratio:.1f}%", delta=f"{pos_ratio-50:.1f}%")
    with col5:
        st.metric("总成交额(亿)", f"{total_vol:.0f}")

    # --- Market Insight ---
    # Calculate sector performance for each period
    sector_perf_1d = df.groupby('板块')['涨跌幅_1d'].mean().sort_values(ascending=False)
    sector_perf_1w = df.groupby('板块')['涨跌幅_1w'].mean().sort_values(ascending=False)
    sector_perf_1m = df.groupby('板块')['涨跌幅_1m'].mean().sort_values(ascending=False)
    
    best_sector_1d = sector_perf_1d.index[0] if not sector_perf_1d.empty else "N/A"
    best_ret_1d = sector_perf_1d.iloc[0] if not sector_perf_1d.empty else 0
    best_sector_1w = sector_perf_1w.index[0] if not sector_perf_1w.empty else "N/A"
    best_ret_1w = sector_perf_1w.iloc[0] if not sector_perf_1w.empty else 0
    best_sector_1m = sector_perf_1m.index[0] if not sector_perf_1m.empty else "N/A"
    best_ret_1m = sector_perf_1m.iloc[0] if not sector_perf_1m.empty else 0
    
    st.markdown(f"""
    <div class="insight-box">
        <strong>📊 市场实时洞察</strong><br>
        最强板块(1日): {best_sector_1d} (+{best_ret_1d:.2f}%) |
        最强板块(1周): {best_sector_1w} (+{best_ret_1w:.2f}%) |
        最强板块(1月): {best_sector_1m} (+{best_ret_1m:.2f}%) |
        上涨家数: {int(pos_ratio*len(df)/100)} / {len(df)}
    </div>
    """, unsafe_allow_html=True)

    # --- Multi-Period Sector Rotation Analysis ---
    st.markdown('<div class="section-header">🔄 多周期板块轮动分析</div>', unsafe_allow_html=True)
    
    # Calculate sector statistics for all periods
    sector_stats = df.groupby('板块').agg({
        '涨跌幅_1d': ['mean', 'std'],
        '涨跌幅_1w': ['mean', 'std'],
        '涨跌幅_1m': ['mean', 'std'],
        '代码': 'count',
        '成交额(亿)': 'sum'
    }).round(2)
    
    sector_stats.columns = [
        '1d_平均涨跌幅', '1d_波动率',
        '1w_平均涨跌幅', '1w_波动率',
        '1m_平均涨跌幅', '1m_波动率',
        '成分股数', '总成交额'
    ]
    sector_stats = sector_stats.reset_index()
    sector_stats['成交额(亿)'] = sector_stats['总成交额'].round(2)
    
    # Create tabs for different periods
    period_tabs = st.tabs(["1日表现", "1周表现", "1月表现"])
    
    for idx, (period, ret_col, vol_col) in enumerate([
        ("1日", "1d_平均涨跌幅", "1d_波动率"),
        ("1周", "1w_平均涨跌幅", "1w_波动率"),
        ("1月", "1m_平均涨跌幅", "1m_波动率")
    ]):
        with period_tabs[idx]:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Bubble chart for this period
                fig = px.scatter(
                    sector_stats,
                    x=ret_col,
                    y='成交额(亿)',
                    size='成分股数',
                    color=ret_col,
                    text='板块',
                    title=f'板块轮动气泡图 ({period}表现)',
                    color_continuous_scale='RdYlGn',
                    size_max=50,
                    labels={ret_col: f'平均涨跌幅 (%)', '成交额(亿)': '成交额 (亿)'}
                )
                fig.update_traces(textposition='top center')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Table for this period
                st.subheader(f"板块详细数据 ({period})")
                display_df = sector_stats[[
                    '板块', ret_col, vol_col, '成分股数', '成交额(亿)'
                ]].copy()
                display_df.columns = [
                    '板块', f'平均涨跌幅(%)', f'波动率', '成分股数', '成交额(亿)'
                ]
                display_df = display_df.sort_values(f'平均涨跌幅(%)', ascending=False)
                
                # Format percentage columns
                display_df_display = display_df.copy()
                display_df_display[f'平均涨跌幅(%)'] = display_df_display[f'平均涨跌幅(%)'].apply(lambda x: f"{x:.2f}%")
                display_df_display[f'波动率'] = display_df_display[f'波动率'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(
                    display_df_display,
                    use_container_width=True,
                    hide_index=True,
                    height=500
                )
    
    # --- Sector Momentum Comparison ---
    st.markdown('<div class="section-header">📊 板块动量对比</div>', unsafe_allow_html=True)
    
    # Prepare data for momentum comparison
    momentum_df = sector_stats[['板块', '1d_平均涨跌幅', '1w_平均涨跌幅', '1m_平均涨跌幅']].copy()
    momentum_df = momentum_df.sort_values('1m_平均涨跌幅', ascending=False).head(15)
    
    # Melt for grouped bar chart
    momentum_melted = momentum_df.melt(
        id_vars=['板块'],
        value_vars=['1d_平均涨跌幅', '1w_平均涨跌幅', '1m_平均涨跌幅'],
        var_name='周期',
        value_name='涨跌幅'
    )
    momentum_melted['周期'] = momentum_melted['周期'].map({
        '1d_平均涨跌幅': '1日',
        '1w_平均涨跌幅': '1周',
        '1m_平均涨跌幅': '1月'
    })
    
    fig = px.bar(
        momentum_melted,
        x='板块',
        y='涨跌幅',
        color='周期',
        barmode='group',
        title='前15板块多周期动量对比',
        labels={'涨跌幅': '涨跌幅 (%)', '板块': ''},
        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- Top Movers ---
    st.markdown('<div class="section-header">📈 全市场龙虎榜</div>', unsafe_allow_html=True)
    
    mover_tabs = st.tabs(["涨幅榜", "跌幅榜", "成交额榜"])
    
    with mover_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1日涨幅前十")
            gain_1d = df.nlargest(10, '涨跌幅_1d')[['代码','名称','板块','最新价','涨跌幅_1d','成交额(亿)']].copy()
            gain_1d['涨跌幅_1d'] = gain_1d['涨跌幅_1d'].apply(lambda x: f"{x:.2f}%")
            st.dataframe(gain_1d, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("1周涨幅前十")
            gain_1w = df.nlargest(10, '涨跌幅_1w')[['代码','名称','板块','最新价','涨跌幅_1w','成交额(亿)']].copy()
            gain_1w['涨跌幅_1w'] = gain_1w['涨跌幅_1w'].apply(lambda x: f"{x:.2f}%")
            st.dataframe(gain_1w, use_container_width=True, hide_index=True)
    
    with mover_tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1日跌幅前十")
            lose_1d = df.nsmallest(10, '涨跌幅_1d')[['代码','名称','板块','最新价','涨跌幅_1d','成交额(亿)']].copy()
            lose_1d['涨跌幅_1d'] = lose_1d['涨跌幅_1d'].apply(lambda x: f"{x:.2f}%")
            st.dataframe(lose_1d, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("1周跌幅前十")
            lose_1w = df.nsmallest(10, '涨跌幅_1w')[['代码','名称','板块','最新价','涨跌幅_1w','成交额(亿)']].copy()
            lose_1w['涨跌幅_1w'] = lose_1w['涨跌幅_1w'].apply(lambda x: f"{x:.2f}%")
            st.dataframe(lose_1w, use_container_width=True, hide_index=True)
    
    with mover_tabs[2]:
        st.subheader("成交额前十")
        volume_top = df.nlargest(10, '成交额(亿)')[['代码','名称','板块','最新价','涨跌幅_1d','成交额(亿)']].copy()
        volume_top['涨跌幅_1d'] = volume_top['涨跌幅_1d'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(volume_top, use_container_width=True, hide_index=True)

    # --- Sector Stock Selector ---
    st.markdown('<div class="section-header">🔍 板块内选股</div>', unsafe_allow_html=True)
    selected_sector = st.selectbox("选择板块", ['全部'] + sorted(df['板块'].unique()))
    if selected_sector != '全部':
        sector_df = df[df['板块'] == selected_sector].copy()
    else:
        sector_df = df.copy()

    # Multi‑factor scoring (using 1d momentum + volume)
    sector_df['动量分'] = (sector_df['涨跌幅_1d'] - sector_df['涨跌幅_1d'].mean()) / sector_df['涨跌幅_1d'].std()
    sector_df['成交额分'] = (sector_df['成交额(亿)'] - sector_df['成交额(亿)'].mean()) / sector_df['成交额(亿)'].std()
    sector_df['综合分'] = (sector_df['动量分']*0.6 + sector_df['成交额分']*0.4).round(2)

    top_stocks = sector_df.nlargest(15, '综合分')[['代码','名称','板块','最新价','涨跌幅_1d','涨跌幅_1w','涨跌幅_1m','成交额(亿)','综合分']]
    top_stocks['涨跌幅_1d'] = top_stocks['涨跌幅_1d'].apply(lambda x: f"{x:.2f}%")
    top_stocks['涨跌幅_1w'] = top_stocks['涨跌幅_1w'].apply(lambda x: f"{x:.2f}%")
    top_stocks['涨跌幅_1m'] = top_stocks['涨跌幅_1m'].apply(lambda x: f"{x:.2f}%")
    st.dataframe(top_stocks, use_container_width=True, hide_index=True)

    # --- Strategy Recommendation ---
    st.markdown('<div class="section-header">📋 实时策略建议</div>', unsafe_allow_html=True)
    
    # Determine market regime using multiple timeframes
    avg_ret_1d = df['涨跌幅_1d'].mean()
    avg_ret_1w = df['涨跌幅_1w'].mean()
    avg_ret_1m = df['涨跌幅_1m'].mean()
    pos_ratio_1d = (df['涨跌幅_1d'] > 0).mean() * 100
    vola_1d = df['涨跌幅_1d'].std()
    
    # Check for trend consistency
    trend_strength = "强势" if avg_ret_1d > 0 and avg_ret_1w > 0 and avg_ret_1m > 0 else "弱势" if avg_ret_1d < 0 and avg_ret_1w < 0 and avg_ret_1m < 0 else "分化"
    
    if avg_ret_1d > 0.5 and pos_ratio_1d > 60:
        regime = "牛市"
        color = "#10B981"
        pos = "70-80%"
        action = "积极进攻"
    elif avg_ret_1d < -0.5 and pos_ratio_1d < 40:
        regime = "熊市"
        color = "#EF4444"
        pos = "20-30%"
        action = "全面防御"
    elif vola_1d > 2:
        regime = "高波动市"
        color = "#F59E0B"
        pos = "40-50%"
        action = "波段操作"
    else:
        regime = "震荡市"
        color = "#4F46E5"
        pos = "50-60%"
        action = "精选个股"

    # Get top and bottom sectors for each period
    top_sectors_1d = sector_perf_1d.head(3).index.tolist()
    top_sectors_1w = sector_perf_1w.head(3).index.tolist()
    top_sectors_1m = sector_perf_1m.head(3).index.tolist()
    
    st.markdown(f"""
    <div class="strategy-box">
        <h3>当前市场状态: <span style="color:{color};">{regime} ({trend_strength})</span></h3>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem;">
            <div><span class="metric-label">建议仓位</span><div class="metric-value">{pos}</div></div>
            <div><span class="metric-label">风险水平</span><div class="metric-value">{'高' if vola_1d>2 else '中' if vola_1d>1 else '低'}</div></div>
            <div><span class="metric-label">操作策略</span><div class="metric-value">{action}</div></div>
        </div>
        <div style="margin-top:1.5rem; border-top:1px solid #ddd; padding-top:1rem;">
            <p><strong>短期强势板块(1日):</strong> {', '.join(top_sectors_1d[:3])}</p>
            <p><strong>中期强势板块(1周):</strong> {', '.join(top_sectors_1w[:3])}</p>
            <p><strong>长期强势板块(1月):</strong> {', '.join(top_sectors_1m[:3])}</p>
            <p><strong>选股参考:</strong> 板块内综合分 > 0 | 成交额 > 板块平均 | 多周期动量向上</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:#666; font-size:0.8rem;">
        数据来源: Yahoo Finance | 成分股列表: universe.csv | 更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
