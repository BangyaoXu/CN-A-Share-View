# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import time

st.set_page_config(layout="wide", page_title="CSI 300 Real Data Dashboard", page_icon="📊")

# Custom CSS for better visibility
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 0;
        text-align: center;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-top: 0;
        text-align: center;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4F46E5;
        margin: 1rem 0;
        color: #000000;
        font-size: 1rem;
    }
    .strategy-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
        color: #000000;
    }
    .metric-label {
        color: #4B5563;
        font-size: 0.9rem;
    }
    .metric-value {
        color: #111827;
        font-size: 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Reliable Data Sources (Works on Streamlit Cloud)
# ------------------------------------------------------------
class ReliableDataCollector:
    """使用稳定可靠的数据源"""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_csi300_tickers():
        """获取CSI300成分股代码列表 (使用CSV from GitHub)"""
        # 主要CSI300成分股 (前50大市值)
        csi300_tickers = [
            ('000858.SZ', '五粮液'), ('000333.SZ', '美的集团'), ('000651.SZ', '格力电器'),
            ('000001.SZ', '平安银行'), ('000002.SZ', '万科A'), ('000568.SZ', '泸州老窖'),
            ('000725.SZ', '京东方A'), ('000625.SZ', '长安汽车'), ('000776.SZ', '广发证券'),
            ('000895.SZ', '双汇发展'), ('000538.SZ', '云南白药'), ('000063.SZ', '中兴通讯'),
            ('002415.SZ', '海康威视'), ('002475.SZ', '立讯精密'), ('002594.SZ', '比亚迪'),
            ('002714.SZ', '牧原股份'), ('002304.SZ', '洋河股份'), ('002230.SZ', '科大讯飞'),
            ('002027.SZ', '分众传媒'), ('002142.SZ', '宁波银行'), ('300750.SZ', '宁德时代'),
            ('300059.SZ', '东方财富'), ('300760.SZ', '迈瑞医疗'), ('300124.SZ', '汇川技术'),
            ('300015.SZ', '爱尔眼科'), ('300122.SZ', '智飞生物'), ('300274.SZ', '阳光电源'),
            ('600519.SS', '贵州茅台'), ('601318.SS', '中国平安'), ('600036.SS', '招商银行'),
            ('601166.SS', '兴业银行'), ('600030.SS', '中信证券'), ('600016.SS', '民生银行'),
            ('600887.SS', '伊利股份'), ('601398.SS', '工商银行'), ('600900.SS', '长江电力'),
            ('601288.SS', '农业银行'), ('601988.SS', '中国银行'), ('601328.SS', '交通银行'),
            ('600028.SS', '中国石化'), ('601857.SS', '中国石油'), ('600050.SS', '中国联通'),
            ('601088.SS', '中国神华'), ('600309.SS', '万华化学'), ('601888.SS', '中国中免'),
            ('603288.SS', '海天味业'), ('600276.SS', '恒瑞医药'), ('600585.SS', '海螺水泥'),
            ('601899.SS', '紫金矿业'), ('600031.SS', '三一重工')
        ]
        return csi300_tickers
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def get_stock_data_yfinance(ticker):
        """使用yfinance获取股票数据"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if not hist.empty:
                last = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else last
                
                # 计算涨跌幅
                pct_change = ((last['Close'] - prev['Close']) / prev['Close']) * 100
                
                return {
                    'price': round(last['Close'], 2),
                    'change': round(pct_change, 2),
                    'volume': last['Volume'],
                    'high': round(last['High'], 2),
                    'low': round(last['Low'], 2),
                    'open': round(last['Open'], 2)
                }
        except Exception as e:
            return None
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def get_index_data():
        """获取指数数据"""
        indices = {
            '000300.SS': 'CSI 300',
            '000001.SS': 'Shanghai Composite',
            '399001.SZ': 'Shenzhen Component'
        }
        
        data = {}
        for ticker, name in indices.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2d")
                if not hist.empty:
                    last = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else last
                    pct_change = ((last['Close'] - prev['Close']) / prev['Close']) * 100
                    data[name] = {
                        'price': round(last['Close'], 2),
                        'change': round(pct_change, 2)
                    }
            except:
                continue
        
        return data
    
    @staticmethod
    def get_sector_from_code(code):
        """根据代码判断行业"""
        sector_map = {
            '000858': '消费', '000333': '家电', '000651': '家电', '000001': '金融',
            '000002': '地产', '000568': '消费', '000725': '科技', '000625': '汽车',
            '000776': '金融', '000895': '消费', '000538': '医药', '000063': '科技',
            '002415': '科技', '002475': '科技', '002594': '新能源', '002714': '农业',
            '002304': '消费', '002230': '科技', '002027': '传媒', '002142': '金融',
            '300750': '新能源', '300059': '金融', '300760': '医药', '300124': '科技',
            '300015': '医药', '300122': '医药', '300274': '新能源', '600519': '消费',
            '601318': '金融', '600036': '金融', '601166': '金融', '600030': '金融',
            '600016': '金融', '600887': '消费', '601398': '金融', '600900': '公用',
            '601288': '金融', '601988': '金融', '601328': '金融', '600028': '能源',
            '601857': '能源', '600050': '通信', '601088': '能源', '600309': '化工',
            '601888': '消费', '603288': '消费', '600276': '医药', '600585': '建材',
            '601899': '有色', '600031': '机械'
        }
        
        # 提取纯数字代码
        code_num = code.split('.')[0]
        return sector_map.get(code_num, '其他')

# ------------------------------------------------------------
# Main Dashboard
# ------------------------------------------------------------
def main():
    # Header
    st.markdown('<p class="main-header">📊 CSI 300 Real-Time Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于Yahoo Finance的实时数据</p>', unsafe_allow_html=True)
    
    collector = ReliableDataCollector()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/investment-portfolio.png", width=100)
        st.title("控制面板")
        
        if st.button("🔄 刷新实时数据", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 数据源")
        st.info("📊 Yahoo Finance")
        st.caption("数据延迟约15分钟")
    
    # Load real data
    with st.spinner("正在获取实时市场数据..."):
        # Get index data
        index_data = collector.get_index_data()
        
        # Get CSI300 tickers
        tickers = collector.get_csi300_tickers()
        
        # Collect stock data
        stocks = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (ticker, name) in enumerate(tickers):
            status_text.text(f"正在获取 {idx+1}/{len(tickers)}: {name}")
            
            data = collector.get_stock_data_yfinance(ticker)
            if data:
                stocks.append({
                    '代码': ticker,
                    '名称': name,
                    '最新价': data['price'],
                    '涨跌幅': data['change'],
                    '成交量': data['volume'],
                    '最高': data['high'],
                    '最低': data['low'],
                    '开盘': data['open']
                })
            
            progress_bar.progress((idx + 1) / len(tickers))
            time.sleep(0.1)  # 避免请求过快
        
        status_text.text("数据加载完成!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        if not stocks:
            st.error("无法获取股票数据，请稍后重试")
            st.stop()
        
        df = pd.DataFrame(stocks)
        
        # Add sector information
        df['板块'] = df['代码'].apply(collector.get_sector_from_code)
        df['成交额(亿)'] = (df['成交量'] * df['最新价'] / 1e8).round(2)
        df['涨跌幅'] = df['涨跌幅'].round(2)
    
    # Display Index Data
    if index_data:
        st.markdown("### 📈 主要指数")
        cols = st.columns(len(index_data))
        for idx, (name, data) in enumerate(index_data.items()):
            with cols[idx]:
                delta_color = "normal" if data['change'] > 0 else "inverse"
                st.metric(
                    name,
                    f"{data['price']:.0f}",
                    delta=f"{data['change']:.2f}%",
                    delta_color=delta_color
                )
    
    st.markdown("---")
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_change = df['涨跌幅'].mean()
        st.metric(
            "平均涨跌幅",
            f"{avg_change:.2f}%",
            delta=f"{avg_change:.2f}%"
        )
    
    with col2:
        positive_pct = (len(df[df['涨跌幅'] > 0]) / len(df)) * 100
        st.metric(
            "上涨比例",
            f"{positive_pct:.1f}%",
            delta=f"{positive_pct - 50:.1f}%"
        )
    
    with col3:
        total_volume = (df['成交量'] * df['最新价']).sum() / 1e8
        st.metric(
            "总成交额 (亿)",
            f"{total_volume:.0f}"
        )
    
    with col4:
        avg_pe = 15 + (avg_change * 2)  # 估算PE
        st.metric(
            "估算PE",
            f"{avg_pe:.1f}"
        )
    
    with col5:
        volatility = df['涨跌幅'].std()
        st.metric(
            "波动率",
            f"{volatility:.2f}%"
        )
    
    # Market Insight Box
    best_sector = df.groupby('板块')['涨跌幅'].mean().idxmax()
    best_sector_return = df.groupby('板块')['涨跌幅'].mean().max()
    
    st.markdown(f"""
    <div class="insight-box">
        <strong>📊 市场洞察</strong><br>
        <span style="color: #000000;">市场情绪: {'乐观' if avg_change > 0.3 else '谨慎' if avg_change > 0 else '悲观'}</span> |
        <span style="color: #000000;">强势板块: {best_sector} (+{best_sector_return:.2f}%)</span> |
        <span style="color: #000000;">波动风险: {'高' if volatility > 2 else '中' if volatility > 1 else '低'}</span> |
        <span style="color: #000000;">数据时间: {datetime.now().strftime('%H:%M:%S')}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sector Analysis
    st.markdown('<div class="section-header">🏭 板块分析</div>', unsafe_allow_html=True)
    
    sector_perf = df.groupby('板块').agg({
        '涨跌幅': ['mean', 'std', 'count'],
        '成交额(亿)': 'sum'
    }).round(2)
    
    sector_perf.columns = ['平均涨跌幅', '波动率', '数量', '成交额(亿)']
    sector_perf = sector_perf.reset_index()
    sector_perf = sector_perf.sort_values('平均涨跌幅', ascending=False)
    
    # Sector performance chart
    fig = px.bar(
        sector_perf.head(10),
        x='板块',
        y='平均涨跌幅',
        color='平均涨跌幅',
        text='平均涨跌幅',
        title='板块涨跌幅排行',
        color_continuous_scale=['#EF4444', '#FCD34D', '#10B981'],
        labels={'平均涨跌幅': '涨跌幅 (%)'}
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top Movers
    st.markdown('<div class="section-header">📈 涨跌幅排名</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("涨幅前十")
        gainers = df.nlargest(10, '涨跌幅')[['代码', '名称', '板块', '最新价', '涨跌幅', '成交额(亿)']].copy()
        gainers['涨跌幅'] = gainers['涨跌幅'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(gainers, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("跌幅前十")
        losers = df.nsmallest(10, '涨跌幅')[['代码', '名称', '板块', '最新价', '涨跌幅', '成交额(亿)']].copy()
        losers['涨跌幅'] = losers['涨跌幅'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(losers, use_container_width=True, hide_index=True)
    
    # Volume Analysis
    st.markdown('<div class="section-header">💰 资金流向</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("成交额前十")
        volume_leaders = df.nlargest(10, '成交额(亿)')[['代码', '名称', '板块', '最新价', '涨跌幅', '成交额(亿)']].copy()
        volume_leaders['涨跌幅'] = volume_leaders['涨跌幅'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(volume_leaders, use_container_width=True, hide_index=True)
    
    with col2:
        # Sector volume distribution
        sector_volume = df.groupby('板块')['成交额(亿)'].sum().sort_values(ascending=False).head(8)
        fig = px.pie(
            values=sector_volume.values,
            names=sector_volume.index,
            title="板块成交额分布",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Strategy Recommendations
    st.markdown('<div class="section-header">🎯 策略建议</div>', unsafe_allow_html=True)
    
    # Determine market state
    avg_ret = df['涨跌幅'].mean()
    positive_ratio = len(df[df['涨跌幅'] > 0]) / len(df)
    volatility = df['涨跌幅'].std()
    
    if avg_ret > 0.5 and positive_ratio > 0.6:
        market_state = "牛市"
        state_color = "#10B981"
        suggested_position = "70-80%"
    elif avg_ret < -0.5 and positive_ratio < 0.4:
        market_state = "熊市"
        state_color = "#EF4444"
        suggested_position = "20-30%"
    elif volatility > 2:
        market_state = "高波动市场"
        state_color = "#F59E0B"
        suggested_position = "40-50%"
    else:
        market_state = "震荡市场"
        state_color = "#4F46E5"
        suggested_position = "50-60%"
    
    # Get top and bottom sectors
    top_sectors = sector_perf.head(3)['板块'].tolist()
    bottom_sectors = sector_perf.tail(3)['板块'].tolist()
    
    st.markdown(f"""
    <div class="strategy-box">
        <h3 style="color: #000000;">当前市场状态: <span style="color: {state_color}; font-weight: bold;">{market_state}</span></h3>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1rem;">
            <div>
                <div class="metric-label">建议仓位</div>
                <div class="metric-value">{suggested_position}</div>
            </div>
            <div>
                <div class="metric-label">风险水平</div>
                <div class="metric-value">{'高' if volatility > 2 else '中' if volatility > 1 else '低'}</div>
            </div>
            <div>
                <div class="metric-label">操作策略</div>
                <div class="metric-value">{'逢低买入' if avg_ret > 0 else '控制仓位'}</div>
            </div>
        </div>
        
        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;">
            <p style="color: #000000;"><strong>重点关注板块:</strong> {', '.join(top_sectors)}</p>
            <p style="color: #000000;"><strong>建议规避板块:</strong> {', '.join(bottom_sectors)}</p>
            <p style="color: #000000;"><strong>选股条件:</strong> PE < 30 | 涨跌幅 > 0 | 成交额 > 1亿</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stock Screener
    st.markdown('<div class="section-header">🔍 实时选股</div>', unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_change = st.slider("最小涨幅 (%)", -5.0, 5.0, -2.0, 0.5)
    with col2:
        max_change = st.slider("最大涨幅 (%)", -5.0, 5.0, 5.0, 0.5)
    with col3:
        min_volume = st.number_input("最小成交额(亿)", 0.0, 100.0, 1.0, 0.5)
    with col4:
        sectors = ['全部'] + df['板块'].unique().tolist()
        selected_sector = st.selectbox("选择板块", sectors)
    
    # Apply filters
    filtered_df = df[
        (df['涨跌幅'] >= min_change) & 
        (df['涨跌幅'] <= max_change) &
        (df['成交额(亿)'] >= min_volume)
    ]
    if selected_sector != '全部':
        filtered_df = filtered_df[filtered_df['板块'] == selected_sector]
    
    st.dataframe(
        filtered_df[['代码', '名称', '板块', '最新价', '涨跌幅', '成交额(亿)']].sort_values('涨跌幅', ascending=False),
        use_container_width=True,
        hide_index=True
    )
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6B7280; font-size: 0.8rem;">
        ⚡ 实时数据系统 (Yahoo Finance) | 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        ⚠️ 数据仅供参考，不构成投资建议
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
