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
    .signal-buy { background-color: #10B981; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: bold; }
    .signal-sell { background-color: #EF4444; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: bold; }
    .signal-neutral { background-color: #F59E0B; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: bold; }
    .fundamental-card { background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb; margin: 0.5rem 0; }
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
    
    end_date = datetime.now()
    start_date_1m = (end_date - timedelta(days=35)).strftime('%Y-%m-%d')
    
    for i, (code, name, sector) in enumerate(ticker_list):
        status.text(f"获取 {i+1}/{total}: {name}")
        yf_ticker = code_to_yf(code)
        try:
            stock = yf.Ticker(yf_ticker)
            hist = stock.history(start=start_date_1m, end=end_date.strftime('%Y-%m-%d'))
            
            if not hist.empty and len(hist) >= 5:
                last = hist.iloc[-1]
                
                # Calculate returns
                ret_1d = ((last['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close']) * 100 if len(hist) >= 2 else 0
                ret_1w = ((last['Close'] - hist.iloc[-5]['Close']) / hist.iloc[-5]['Close']) * 100 if len(hist) >= 5 else ret_1d
                
                if len(hist) >= 20:
                    ret_1m = ((last['Close'] - hist.iloc[-20]['Close']) / hist.iloc[-20]['Close']) * 100
                elif len(hist) >= 10:
                    ret_1m = ((last['Close'] - hist.iloc[0]['Close']) / hist.iloc[0]['Close']) * 100
                else:
                    ret_1m = ret_1w
                
                stocks.append({
                    '代码': code,
                    '名称': name,
                    '板块': sector,
                    'yf_ticker': yf_ticker,
                    '最新价': round(last['Close'], 2),
                    '涨跌幅_1d': round(ret_1d, 2),
                    '涨跌幅_1w': round(ret_1w, 2),
                    '涨跌幅_1m': round(ret_1m, 2),
                    '成交量': last['Volume'],
                    '成交额(亿)': round(last['Volume'] * last['Close'] / 1e8, 2),
                })
        except Exception as e:
            pass
        prog.progress((i+1)/total)
        time.sleep(0.1)
    status.empty()
    prog.empty()
    return pd.DataFrame(stocks)

# ------------------------------------------------------------
# Technical Analysis Functions
# ------------------------------------------------------------
def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    df = df.copy()
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    return df

def calculate_kdj(df, period=9, k_smooth=3, d_smooth=3):
    """Calculate KDJ indicator"""
    df = df.copy()
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    
    df['RSV'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['K'] = df['RSV'].ewm(span=k_smooth, adjust=False).mean()
    df['D'] = df['K'].ewm(span=d_smooth, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df

def calculate_rsi(df, period=14):
    """Calculate RSI indicator"""
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    df = df.copy()
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=period).mean()
    df['ATR_pct'] = (df['ATR'] / df['Close']) * 100
    return df

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    df = df.copy()
    df['BB_Middle'] = df['Close'].rolling(window=period).mean()
    bb_std = df['Close'].rolling(window=period).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * std_dev)
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']) * 100
    df['BB_Position'] = ((df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])) * 100
    return df

def calculate_all_indicators(df):
    """Calculate all technical indicators"""
    df = calculate_macd(df)
    df = calculate_kdj(df)
    df = calculate_rsi(df)
    df = calculate_atr(df)
    df = calculate_bollinger_bands(df)
    return df

# ------------------------------------------------------------
# Generate trading signals based on indicators
# ------------------------------------------------------------
def generate_trading_signals(latest):
    """Generate trading signals based on latest indicator values"""
    signals = []
    score = 0
    
    # MACD Signal
    if latest['MACD'] > latest['Signal'] and latest['MACD_Hist'] > 0:
        signals.append(("MACD", "买入", 2))
        score += 2
    elif latest['MACD'] < latest['Signal'] and latest['MACD_Hist'] < 0:
        signals.append(("MACD", "卖出", -2))
        score -= 2
    else:
        signals.append(("MACD", "中性", 0))
    
    # KDJ Signal
    if latest['K'] > latest['D'] and latest['J'] > latest['K'] and latest['K'] < 30:
        signals.append(("KDJ", "超卖反弹", 2))
        score += 2
    elif latest['K'] < latest['D'] and latest['J'] < latest['K'] and latest['K'] > 70:
        signals.append(("KDJ", "超买回调", -2))
        score -= 2
    elif latest['K'] > latest['D']:
        signals.append(("KDJ", "金叉", 1))
        score += 1
    elif latest['K'] < latest['D']:
        signals.append(("KDJ", "死叉", -1))
        score -= 1
    else:
        signals.append(("KDJ", "中性", 0))
    
    # RSI Signal
    if latest['RSI'] < 30:
        signals.append(("RSI", "超卖", 2))
        score += 2
    elif latest['RSI'] > 70:
        signals.append(("RSI", "超买", -2))
        score -= 2
    elif 40 <= latest['RSI'] <= 60:
        signals.append(("RSI", "中性", 0))
    elif latest['RSI'] > 50:
        signals.append(("RSI", "偏强", 1))
        score += 1
    else:
        signals.append(("RSI", "偏弱", -1))
        score -= 1
    
    # Bollinger Bands Signal
    if latest['Close'] < latest['BB_Lower']:
        signals.append(("Bollinger", "下轨支撑", 2))
        score += 2
    elif latest['Close'] > latest['BB_Upper']:
        signals.append(("Bollinger", "上轨压力", -2))
        score -= 2
    elif latest['Close'] < latest['BB_Middle']:
        signals.append(("Bollinger", "中轨下方", -1))
        score -= 1
    elif latest['Close'] > latest['BB_Middle']:
        signals.append(("Bollinger", "中轨上方", 1))
        score += 1
    else:
        signals.append(("Bollinger", "中性", 0))
    
    # Overall signal
    if score >= 4:
        overall = ("整体", "强烈买入", score, "#10B981")
    elif score >= 2:
        overall = ("整体", "买入", score, "#6EE7B7")
    elif score <= -4:
        overall = ("整体", "强烈卖出", score, "#EF4444")
    elif score <= -2:
        overall = ("整体", "卖出", score, "#FCA5A5")
    else:
        overall = ("整体", "观望", score, "#F59E0B")
    
    return signals, overall

# ------------------------------------------------------------
# Get fundamental data from Yahoo Finance
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_fundamental_data(yf_ticker):
    """Get fundamental data for a stock"""
    try:
        stock = yf.Ticker(yf_ticker)
        info = stock.info
        
        # Earnings history and estimates
        earnings_dates = []
        try:
            earnings = stock.earnings_dates
            if earnings is not None and not earnings.empty:
                earnings_dates = earnings.head(8).reset_index()
        except:
            pass
        
        # Analyst recommendations
        recommendations = []
        try:
            rec = stock.recommendations
            if rec is not None and not rec.empty:
                recommendations = rec.tail(5).reset_index()
        except:
            pass
        
        # Key statistics
        fundamentals = {
            '市值(亿)': info.get('marketCap', 0) / 1e8 if info.get('marketCap') else 0,
            'PE(TTM)': info.get('trailingPE', 0),
            'PE(滚动)': info.get('forwardPE', 0),
            'PEG': info.get('pegRatio', 0),
            'PB': info.get('priceToBook', 0),
            'PS': info.get('priceToSalesTrailing12Months', 0),
            '股息率(%)': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'ROE(%)': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
            'ROA(%)': info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0,
            '毛利率(%)': info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 0,
            '净利率(%)': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
            '营收增长(%)': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
            '每股收益': info.get('trailingEps', 0),
            '每股净资产': info.get('bookValue', 0),
            '资产负债率(%)': info.get('debtToEquity', 0) if info.get('debtToEquity') else 0,
            'Beta': info.get('beta', 0),
            '52周高点': info.get('fiftyTwoWeekHigh', 0),
            '52周低点': info.get('fiftyTwoWeekLow', 0),
            '平均成交量': info.get('averageVolume', 0),
        }
        
        return fundamentals, earnings_dates, recommendations, info
    except Exception as e:
        return None, None, None, None

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
    st.markdown('<p class="main-header">📊 CSI 800 + CSI 1000 仪表盘</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">实时行情 + 技术分析 + 基本面深度挖掘 + 多周期板块轮动</p>', unsafe_allow_html=True)

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
        st.info("📉 技术指标: MACD, KDJ, RSI, ATR, Bollinger")
        st.info("📚 基本面: Yahoo Finance")
        st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    constituents = load_constituents()
    ticker_list = list(zip(constituents['code'], constituents['name'], constituents['sector']))

    with st.spinner("获取实时行情..."):
        df = fetch_realtime_stocks(ticker_list)

    if df.empty:
        st.error("未能获取任何股票数据，请检查网络")
        st.stop()

    # --- Stock Selection for Deep Dive ---
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 个股深度研究</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        selected_stock = st.selectbox(
            "选择股票",
            options=df['代码'] + ' - ' + df['名称'],
            format_func=lambda x: x
        )
        selected_code = selected_stock.split(' - ')[0]
        stock_info = df[df['代码'] == selected_code].iloc[0]
    
    with col2:
        period = st.selectbox(
            "分析周期",
            options=["1mo", "3mo", "6mo", "1y", "2y"],
            index=2
        )
    
    with col3:
        st.metric(
            stock_info['名称'],
            f"{stock_info['最新价']:.2f}",
            delta=f"{stock_info['涨跌幅_1d']:.2f}%"
        )

    # --- Technical Analysis Section ---
    if selected_code:
        yf_ticker = code_to_yf(selected_code)
        
        with st.spinner("加载技术指标..."):
            # Fetch historical data
            stock = yf.Ticker(yf_ticker)
            hist = stock.history(period=period)
            
            if not hist.empty:
                # Calculate all indicators
                hist_with_indicators = calculate_all_indicators(hist)
                latest = hist_with_indicators.iloc[-1]
                
                # Generate trading signals
                signals, overall = generate_trading_signals(latest)
                
                # Display overall signal
                signal_color = overall[3]
                st.markdown(f"""
                <div style="background-color: {signal_color}20; padding: 1rem; border-radius: 10px; border-left: 4px solid {signal_color}; margin: 1rem 0;">
                    <h3 style="color: {signal_color}; margin: 0;">{overall[1]} (信号强度: {overall[2]})</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Create price chart with indicators
                fig = make_subplots(
                    rows=5, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
                    subplot_titles=('价格 & 技术指标', 'MACD', 'KDJ', 'RSI', 'ATR & Bollinger Width')
                )
                
                # Price and Bollinger Bands
                fig.add_trace(go.Candlestick(
                    x=hist_with_indicators.index,
                    open=hist_with_indicators['Open'],
                    high=hist_with_indicators['High'],
                    low=hist_with_indicators['Low'],
                    close=hist_with_indicators['Close'],
                    name='K线'
                ), row=1, col=1)
                
                # Bollinger Bands
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['BB_Upper'],
                    line=dict(color='rgba(250, 128, 114, 0.5)', width=1),
                    name='上轨'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['BB_Lower'],
                    line=dict(color='rgba(250, 128, 114, 0.5)', width=1),
                    name='下轨',
                    fill='tonexty',
                    fillcolor='rgba(250, 128, 114, 0.1)'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['BB_Middle'],
                    line=dict(color='orange', width=1, dash='dash'),
                    name='中轨'
                ), row=1, col=1)
                
                # Volume
                colors = ['red' if hist_with_indicators['Close'].iloc[i] >= hist_with_indicators['Open'].iloc[i] 
                         else 'green' for i in range(len(hist_with_indicators))]
                
                fig.add_trace(go.Bar(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['Volume'],
                    name='成交量',
                    marker_color=colors,
                    opacity=0.5
                ), row=1, col=1, secondary_y=False)
                
                # MACD
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['MACD'],
                    line=dict(color='blue', width=2),
                    name='MACD'
                ), row=2, col=1)
                
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['Signal'],
                    line=dict(color='red', width=2),
                    name='Signal'
                ), row=2, col=1)
                
                # MACD Histogram
                colors_macd = ['red' if x < 0 else 'green' for x in hist_with_indicators['MACD_Hist']]
                fig.add_trace(go.Bar(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['MACD_Hist'],
                    name='MACD Hist',
                    marker_color=colors_macd,
                    opacity=0.5
                ), row=2, col=1)
                
                # KDJ
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['K'],
                    line=dict(color='blue', width=2),
                    name='K'
                ), row=3, col=1)
                
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['D'],
                    line=dict(color='red', width=2),
                    name='D'
                ), row=3, col=1)
                
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['J'],
                    line=dict(color='green', width=2),
                    name='J'
                ), row=3, col=1)
                
                # RSI with levels
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['RSI'],
                    line=dict(color='purple', width=2),
                    name='RSI'
                ), row=4, col=1)
                
                # RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=4, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=4, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=4, col=1)
                
                # ATR and Bollinger Width
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['ATR_pct'],
                    line=dict(color='brown', width=2),
                    name='ATR%'
                ), row=5, col=1)
                
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['BB_Width'],
                    line=dict(color='orange', width=2, dash='dash'),
                    name='BB宽度%'
                ), row=5, col=1)
                
                # Update layout
                fig.update_layout(
                    height=1200,
                    showlegend=True,
                    hovermode='x unified',
                    title=f"{stock_info['名称']} ({selected_code}) - 技术分析"
                )
                
                fig.update_xatches(rangeslider_visible=False)
                fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Signal Table
                st.markdown("### 📊 技术信号汇总")
                signal_df = pd.DataFrame(signals, columns=['指标', '信号', '强度'])
                signal_df['强度'] = signal_df['强度'].map({2: '++', 1: '+', 0: '○', -1: '-', -2: '--'})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(signal_df, use_container_width=True, hide_index=True)
                
                with col2:
                    # Latest indicator values
                    latest_df = pd.DataFrame({
                        '指标': ['MACD', '信号线', 'K值', 'D值', 'J值', 'RSI', 'ATR%', '布林带宽%', '布林位置%'],
                        '数值': [
                            f"{latest['MACD']:.2f}",
                            f"{latest['Signal']:.2f}",
                            f"{latest['K']:.2f}",
                            f"{latest['D']:.2f}",
                            f"{latest['J']:.2f}",
                            f"{latest['RSI']:.2f}",
                            f"{latest['ATR_pct']:.2f}%",
                            f"{latest['BB_Width']:.2f}%",
                            f"{latest['BB_Position']:.2f}%"
                        ]
                    })
                    st.dataframe(latest_df, use_container_width=True, hide_index=True)

    # --- Fundamental Analysis Section ---
    st.markdown('<div class="section-header">📚 基本面深度分析</div>', unsafe_allow_html=True)
    
    if selected_code:
        with st.spinner("加载基本面数据..."):
            yf_ticker = code_to_yf(selected_code)
            fundamentals, earnings_dates, recommendations, info = get_fundamental_data(yf_ticker)
            
            if fundamentals:
                # Key Metrics in columns
                st.markdown("### 关键指标")
                cols = st.columns(4)
                metrics = [
                    ('市值(亿)', f"{fundamentals['市值(亿)']:.0f}"),
                    ('PE(TTM)', f"{fundamentals['PE(TTM)']:.2f}" if fundamentals['PE(TTM)'] > 0 else 'N/A'),
                    ('PB', f"{fundamentals['PB']:.2f}" if fundamentals['PB'] > 0 else 'N/A'),
                    ('ROE(%)', f"{fundamentals['ROE(%)']:.1f}%" if fundamentals['ROE(%)'] > 0 else 'N/A'),
                ]
                
                for idx, (label, value) in enumerate(metrics):
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="fundamental-card">
                            <div style="color: #666; font-size: 0.9rem;">{label}</div>
                            <div style="font-size: 1.5rem; font-weight: bold;">{value}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Valuation & Profitability
                st.markdown("### 💰 估值与盈利能力")
                col1, col2 = st.columns(2)
                
                with col1:
                    valuation_df = pd.DataFrame({
                        '指标': ['PE(滚动)', 'PEG', 'PS', '股息率(%)', 'Beta'],
                        '数值': [
                            f"{fundamentals['PE(滚动)']:.2f}" if fundamentals['PE(滚动)'] > 0 else 'N/A',
                            f"{fundamentals['PEG']:.2f}" if fundamentals['PEG'] > 0 else 'N/A',
                            f"{fundamentals['PS']:.2f}" if fundamentals['PS'] > 0 else 'N/A',
                            f"{fundamentals['股息率(%)']:.2f}%" if fundamentals['股息率(%)'] > 0 else 'N/A',
                            f"{fundamentals['Beta']:.2f}" if fundamentals['Beta'] > 0 else 'N/A'
                        ]
                    })
                    st.dataframe(valuation_df, use_container_width=True, hide_index=True)
                
                with col2:
                    profitability_df = pd.DataFrame({
                        '指标': ['毛利率(%)', '净利率(%)', 'ROA(%)', '营收增长(%)', '资产负债率(%)'],
                        '数值': [
                            f"{fundamentals['毛利率(%)']:.1f}%" if fundamentals['毛利率(%)'] > 0 else 'N/A',
                            f"{fundamentals['净利率(%)']:.1f}%" if fundamentals['净利率(%)'] > 0 else 'N/A',
                            f"{fundamentals['ROA(%)']:.1f}%" if fundamentals['ROA(%)'] > 0 else 'N/A',
                            f"{fundamentals['营收增长(%)']:.1f}%" if fundamentals['营收增长(%)'] > 0 else 'N/A',
                            f"{fundamentals['资产负债率(%)']:.1f}%" if fundamentals['资产负债率(%)'] > 0 else 'N/A'
                        ]
                    })
                    st.dataframe(profitability_df, use_container_width=True, hide_index=True)
                
                # Price Levels
                st.markdown("### 📈 价格水平")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    current = stock_info['最新价']
                    high_52w = fundamentals['52周高点']
                    low_52w = fundamentals['52周低点']
                    
                    if high_52w > 0:
                        from_high = ((high_52w - current) / high_52w) * 100
                        from_low = ((current - low_52w) / low_52w) * 100
                        
                        st.markdown(f"""
                        <div class="fundamental-card">
                            <div>当前价格: <b>{current:.2f}</b></div>
                            <div>52周高点: {high_52w:.2f} (距离 {from_high:.1f}%)</div>
                            <div>52周低点: {low_52w:.2f} (距离 {from_low:.1f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Analyst recommendations
                    if recommendations is not None and not recommendations.empty:
                        st.markdown("#### 分析师评级")
                        st.dataframe(recommendations.head(), use_container_width=True)
                
                with col3:
                    # Earnings dates
                    if earnings_dates is not None and not earnings_dates.empty:
                        st.markdown("#### 财报日期")
                        st.dataframe(earnings_dates[['Date', 'EPS']].head(), use_container_width=True)
                
                # Business Summary
                if info and 'longBusinessSummary' in info:
                    with st.expander("公司业务概览"):
                        st.write(info['longBusinessSummary'])

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
    sector_perf_1d = df.groupby('板块')['涨跌幅_1d'].mean().sort_values(ascending=False)
    sector_perf_1w = df.groupby('板块')['涨跌幅_1w'].mean().sort_values(ascending=False)
    sector_perf_1m = df.groupby('板块')['涨跌幅_1m'].mean().sort_values(ascending=False)
    
    best_sector_1d = sector_perf_1d.index[0] if not sector_perf_1d.empty else "N/A"
    best_ret_1d = sector_perf_1d.iloc[0] if not sector_perf_1d.empty else 0
    best_sector_1m = sector_perf_1m.index[0] if not sector_perf_1m.empty else "N/A"
    best_ret_1m = sector_perf_1m.iloc[0] if not sector_perf_1m.empty else 0
    
    st.markdown(f"""
    <div class="insight-box">
        <strong>📊 市场实时洞察</strong><br>
        短期强势: {best_sector_1d} (+{best_ret_1d:.2f}%) |
        长期强势: {best_sector_1m} (+{best_ret_1m:.2f}%) |
        上涨家数: {int(pos_ratio*len(df)/100)} / {len(df)}
    </div>
    """, unsafe_allow_html=True)

    # --- Multi-Period Sector Rotation Analysis ---
    st.markdown('<div class="section-header">🔄 多周期板块轮动分析</div>', unsafe_allow_html=True)
    
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
    
    period_tabs = st.tabs(["1日表现", "1周表现", "1月表现"])
    
    for idx, (period, ret_col) in enumerate([
        ("1日", "1d_平均涨跌幅"),
        ("1周", "1w_平均涨跌幅"),
        ("1月", "1m_平均涨跌幅")
    ]):
        with period_tabs[idx]:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig = px.scatter(
                    sector_stats,
                    x=ret_col,
                    y='成交额(亿)',
                    size='成分股数',
                    color=ret_col,
                    text='板块',
                    title=f'板块轮动气泡图 ({period}表现)',
                    color_continuous_scale='RdYlGn',
                    size_max=50
                )
                fig.update_traces(textposition='top center')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                display_df = sector_stats[[
                    '板块', ret_col, f'{ret_col.split("_")[0]}_波动率', '成分股数', '成交额(亿)'
                ]].copy()
                display_df.columns = ['板块', '平均涨跌幅(%)', '波动率', '成分股数', '成交额(亿)']
                display_df = display_df.sort_values('平均涨跌幅(%)', ascending=False)
                
                display_df_display = display_df.copy()
                display_df_display['平均涨跌幅(%)'] = display_df_display['平均涨跌幅(%)'].apply(lambda x: f"{x:.2f}%")
                display_df_display['波动率'] = display_df_display['波动率'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(
                    display_df_display,
                    use_container_width=True,
                    hide_index=True,
                    height=500
                )

    # --- Sector Momentum Comparison ---
    st.markdown('<div class="section-header">📊 板块动量对比</div>', unsafe_allow_html=True)
    
    momentum_df = sector_stats[['板块', '1d_平均涨跌幅', '1w_平均涨跌幅', '1m_平均涨跌幅']].copy()
    momentum_df = momentum_df.sort_values('1m_平均涨跌幅', ascending=False).head(15)
    
    momentum_melted = momentum_df.melt(
        id_vars=['板块'],
        value_vars=['1d_平均涨跌幅', '1w_平均涨跌幅', '1m_平均涨跌幅'],
        var_name='周期',
        value_name='涨跌幅'
    )
    momentum_melted['周期'] = momentum_melted['周期'].map({
        '1d_平均涨跌幅': '1日',
        '1w_平均涨跌幅': '1周',
        '1月_平均涨跌幅': '1月'
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
    selected_sector = st.selectbox("选择板块", ['全部'] + sorted(df['板块'].unique()), key='sector_selector')
    if selected_sector != '全部':
        sector_df = df[df['板块'] == selected_sector].copy()
    else:
        sector_df = df.copy()

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
    
    avg_ret_1d = df['涨跌幅_1d'].mean()
    avg_ret_1w = df['涨跌幅_1w'].mean()
    avg_ret_1m = df['涨跌幅_1m'].mean()
    pos_ratio_1d = (df['涨跌幅_1d'] > 0).mean() * 100
    vola_1d = df['涨跌幅_1d'].std()
    
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
        数据来源: Yahoo Finance | 成分股列表: universe.csv | 对冲基金级分析系统 <br>
        技术指标: MACD(12,26,9) | KDJ(9,3,3) | RSI(14) | ATR(14) | Bollinger Bands(20,2)<br>
        更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
