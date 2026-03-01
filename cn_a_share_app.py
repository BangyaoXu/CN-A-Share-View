# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import requests
import json

# Attempt to import akshare
try:
    import akshare as ak
except ImportError:
    st.error("请先安装 akshare：pip install akshare")
    st.stop()

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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .warning-card {
        background: #FEF3C7;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #F59E0B;
        color: #000000;
    }
    .signal-green {
        color: #10B981;
        font-weight: bold;
    }
    .signal-red {
        color: #EF4444;
        font-weight: bold;
    }
    .signal-yellow {
        color: #F59E0B;
        font-weight: bold;
    }
    .hedge-fund-badge {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
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
    .stAlert {
        color: #000000;
    }
    p, li, span, div {
        color: #000000;
    }
    .stMarkdown {
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Real Data Collection Functions
# ------------------------------------------------------------
class RealDataCollector:
    """Collect REAL market data from multiple sources"""
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def get_realtime_market_data():
        """获取实时市场数据"""
        try:
            # Get real-time quotes for all A-shares
            df = ak.stock_zh_a_spot_em()
            if not df.empty:
                return df
        except Exception as e:
            st.error(f"实时数据获取失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_north_flow():
        """获取北向资金数据"""
        try:
            df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
            if not df.empty:
                return df
        except:
            pass
        return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_margin_data():
        """获取融资融券数据"""
        try:
            df = ak.stock_margin_sse()
            if not df.empty:
                return df
        except:
            pass
        return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=86400)
    def get_csi300_constituents():
        """获取沪深300成分股"""
        try:
            # Try multiple sources
            sources = [
                lambda: ak.index_stock_cons_csindex("000300"),
                lambda: ak.index_stock_cons(symbol="000300")
            ]
            
            for source in sources:
                try:
                    df = source()
                    if df is not None and not df.empty:
                        return df
                except:
                    continue
        except:
            pass
        
        st.error("无法获取沪深300成分股数据")
        return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_stock_quote(code):
        """获取单只股票实时行情"""
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", 
                                   start_date=(datetime.now() - timedelta(days=5)).strftime('%Y%m%d'),
                                   end_date=datetime.now().strftime('%Y%m%d'),
                                   adjust="qfq")
            if not df.empty:
                return df.iloc[-1]
        except:
            pass
        return None

# ------------------------------------------------------------
# Main Dashboard
# ------------------------------------------------------------
def main():
    # Header
    st.markdown('<p class="main-header">📊 CSI 300 Real-Time Trading Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于真实市场数据的量化分析系统</p>', unsafe_allow_html=True)
    
    collector = RealDataCollector()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/investment-portfolio.png", width=100)
        st.title("控制面板")
        
        if st.button("🔄 刷新实时数据", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 数据状态")
        st.info("📡 实时数据源: AkShare")
    
    # Load real data
    with st.spinner("正在获取实时市场数据..."):
        # Get CSI300 constituents
        constituents_df = collector.get_csi300_constituents()
        
        if constituents_df.empty:
            st.error("无法获取成分股数据，请检查网络连接")
            st.stop()
        
        # Display column info for debugging
        with st.expander("数据源信息"):
            st.write("找到以下数据列:", constituents_df.columns.tolist())
        
        # Identify code and name columns
        code_col = None
        name_col = None
        
        for col in constituents_df.columns:
            if '代码' in col or 'code' in col.lower():
                code_col = col
            if '名称' in col or 'name' in col.lower():
                name_col = col
        
        if not code_col:
            code_col = constituents_df.columns[0]
        if not name_col:
            name_col = constituents_df.columns[1] if len(constituents_df.columns) > 1 else constituents_df.columns[0]
        
        # Get real market data
        market_data = collector.get_realtime_market_data()
        
        # Get north flow data
        north_flow_df = collector.get_north_flow()
        north_flow_value = north_flow_df['value'].iloc[-1] / 1e8 if not north_flow_df.empty else 0
        
        # Get margin data
        margin_df = collector.get_margin_data()
        margin_value = margin_df['融资余额'].iloc[-1] / 1e8 if not margin_df.empty else 0
        
        # Process stock data
        stocks = []
        total_stocks = min(50, len(constituents_df))  # Limit to 50 for performance
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in constituents_df.head(total_stocks).iterrows():
            status_text.text(f"正在获取 {idx+1}/{total_stocks} 只股票数据...")
            
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip()
            
            # Clean code
            code = ''.join(filter(str.isdigit, code))
            if len(code) < 6:
                code = code.zfill(6)
            
            # Get real quote
            quote = collector.get_stock_quote(code)
            
            if quote is not None:
                stocks.append({
                    '代码': code,
                    '名称': name,
                    '最新价': quote['收盘'],
                    '涨跌幅': quote['涨跌幅'],
                    '成交量': quote['成交量'],
                    '成交额': quote['成交额'],
                    '最高': quote['最高'],
                    '最低': quote['最低'],
                    '开盘': quote['开盘']
                })
            
            progress_bar.progress((idx + 1) / total_stocks)
        
        status_text.text("数据加载完成!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        if not stocks:
            st.error("无法获取任何股票实时数据")
            st.stop()
        
        df = pd.DataFrame(stocks)
        
        # Add sector information based on real industry classification
        def get_sector(code):
            try:
                info = ak.stock_individual_info_em(symbol=code)
                if not info.empty:
                    sector_row = info[info['item'] == '行业']
                    if not sector_row.empty:
                        return sector_row['value'].iloc[0]
            except:
                pass
            
            # Fallback to code-based classification
            code_prefix = code[:3]
            sector_map = {
                '600': '制造业', '601': '金融', '603': '制造业',
                '000': '综合', '001': '综合', '002': '中小板',
                '300': '创业板', '688': '科创板'
            }
            return sector_map.get(code_prefix, '其他')
        
        df['板块'] = df['代码'].apply(get_sector)
        df['成交额(亿)'] = (df['成交额'] / 1e8).round(2)
    
    # Key Metrics
    st.markdown("---")
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
        st.metric(
            "北向资金 (亿)",
            f"{north_flow_value:.1f}",
            delta="流入" if north_flow_value > 0 else "流出"
        )
    
    with col4:
        st.metric(
            "融资余额 (亿)",
            f"{margin_value:.0f}"
        )
    
    with col5:
        total_volume = df['成交额'].sum() / 1e8
        st.metric(
            "总成交额 (亿)",
            f"{total_volume:.0f}"
        )
    
    # Market Insight Box - Fixed visibility
    st.markdown(f"""
    <div class="insight-box">
        <strong>📊 市场洞察</strong><br>
        <span style="color: #000000;">市场情绪: {'乐观' if avg_change > 0.5 else '谨慎' if avg_change > 0 else '悲观'}</span> |
        <span style="color: #000000;">北向资金: {'净流入' if north_flow_value > 0 else '净流出'}</span> |
        <span style="color: #000000;">强势板块: {df.groupby('板块')['涨跌幅'].mean().idxmax()} (+{df.groupby('板块')['涨跌幅'].mean().max():.2f}%)</span> |
        <span style="color: #000000;">波动风险: {'高' if df['涨跌幅'].std() > 2 else '中' if df['涨跌幅'].std() > 1 else '低'}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sector Analysis
    st.markdown('<div class="section-header">🏭 板块分析</div>', unsafe_allow_html=True)
    
    sector_perf = df.groupby('板块').agg({
        '涨跌幅': ['mean', 'std', 'count'],
        '成交额': 'sum'
    }).round(2)
    
    sector_perf.columns = ['平均涨跌幅', '波动率', '数量', '成交额']
    sector_perf = sector_perf.reset_index()
    sector_perf['成交额(亿)'] = (sector_perf['成交额'] / 1e8).round(0)
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
        gainers = df.nlargest(10, '涨跌幅')[['代码', '名称', '板块', '涨跌幅', '成交额(亿)']].copy()
        gainers['涨跌幅'] = gainers['涨跌幅'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(gainers, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("跌幅前十")
        losers = df.nsmallest(10, '涨跌幅')[['代码', '名称', '板块', '涨跌幅', '成交额(亿)']].copy()
        losers['涨跌幅'] = losers['涨跌幅'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(losers, use_container_width=True, hide_index=True)
    
    # Volume Analysis
    st.markdown('<div class="section-header">💰 资金流向</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("成交额前十")
        volume_leaders = df.nlargest(10, '成交额')[['代码', '名称', '板块', '涨跌幅', '成交额(亿)']].copy()
        volume_leaders['涨跌幅'] = volume_leaders['涨跌幅'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(volume_leaders, use_container_width=True, hide_index=True)
    
    with col2:
        # Sector volume distribution
        sector_volume = df.groupby('板块')['成交额'].sum().sort_values(ascending=False).head(8)
        fig = px.pie(
            values=sector_volume.values,
            names=sector_volume.index,
            title="板块成交额分布",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Strategy Recommendations - Fixed visibility
    st.markdown('<div class="section-header">🎯 策略建议</div>', unsafe_allow_html=True)
    
    # Determine market state
    avg_ret = df['涨跌幅'].mean()
    positive_ratio = len(df[df['涨跌幅'] > 0]) / len(df)
    volatility = df['涨跌幅'].std()
    
    if avg_ret > 0.5 and positive_ratio > 0.6:
        market_state = "牛市"
        state_color = "#10B981"
    elif avg_ret < -0.5 and positive_ratio < 0.4:
        market_state = "熊市"
        state_color = "#EF4444"
    elif volatility > 2:
        market_state = "高波动市场"
        state_color = "#F59E0B"
    else:
        market_state = "震荡市场"
        state_color = "#4F46E5"
    
    # Generate strategy based on real data
    if avg_ret > 0.5:
        strategy = "逢低买入强势板块"
        risk_level = "中等"
    elif avg_ret < -0.5:
        strategy = "控制仓位，等待企稳"
        risk_level = "高"
    else:
        strategy = "均衡配置，精选个股"
        risk_level = "中等"
    
    st.markdown(f"""
    <div class="strategy-box">
        <h3 style="color: #000000;">当前市场状态: <span style="color: {state_color}; font-weight: bold;">{market_state}</span></h3>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1rem;">
            <div>
                <div class="metric-label">建议仓位</div>
                <div class="metric-value">{'70%' if avg_ret > 0.5 else '30%' if avg_ret < -0.5 else '50%'}</div>
            </div>
            <div>
                <div class="metric-label">风险水平</div>
                <div class="metric-value">{risk_level}</div>
            </div>
            <div>
                <div class="metric-label">操作策略</div>
                <div class="metric-value">{strategy}</div>
            </div>
        </div>
        
        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;">
            <p style="color: #000000;"><strong>重点关注板块:</strong> {', '.join(sector_perf.head(3)['板块'].tolist())}</p>
            <p style="color: #000000;"><strong>建议规避板块:</strong> {', '.join(sector_perf.tail(3)['板块'].tolist())}</p>
            <p style="color: #000000;"><strong>止损建议:</strong> 跌破5日均线减仓，跌破10日均线清仓</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stock Screener
    st.markdown('<div class="section-header">🔍 实时选股</div>', unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_change = st.slider("最小涨幅 (%)", -5.0, 5.0, -2.0, 0.5)
    with col2:
        max_change = st.slider("最大涨幅 (%)", -5.0, 5.0, 2.0, 0.5)
    with col3:
        sectors = ['全部'] + df['板块'].unique().tolist()
        selected_sector = st.selectbox("选择板块", sectors)
    
    # Apply filters
    filtered_df = df[(df['涨跌幅'] >= min_change) & (df['涨跌幅'] <= max_change)]
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
        ⚡ 实时数据系统 | 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        ⚠️ 数据仅供参考，不构成投资建议
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
