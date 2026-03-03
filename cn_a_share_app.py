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
    .fundamental-card { 
        background-color: #ffffff; 
        padding: 1rem; 
        border-radius: 8px; 
        border: 1px solid #e5e7eb; 
        margin: 0.5rem 0; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .fundamental-card div { 
        color: #000000 !important; 
    }
    .fundamental-card .metric-label { 
        color: #4B5563 !important; 
        font-size: 0.9rem; 
    }
    .fundamental-card .metric-value { 
        color: #111827 !important; 
        font-size: 1.5rem; 
        font-weight: bold; 
    }
    .price-up { color: #EF4444; font-weight: bold; }
    .price-down { color: #10B981; font-weight: bold; }
    .data-warning { background-color: #fff3cd; color: #856404; padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0; }
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
# Yahoo Finance helper with retry logic
# ------------------------------------------------------------
def code_to_yf(code):
    code = str(code).zfill(6)
    return f"{code}.SS" if code.startswith(('5','6')) else f"{code}.SZ"

# ------------------------------------------------------------
# Fetch real-time stock data with rate limiting (cached 30 min)
# ------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_realtime_stocks(ticker_list):
    stocks = []
    prog = st.progress(0)
    status = st.empty()
    total = len(ticker_list)
    
    end_date = datetime.now()
    start_date_3m = (end_date - timedelta(days=95)).strftime('%Y-%m-%d')  # Get 3 months for better 1m calculation
    
    for i, (code, name, sector) in enumerate(ticker_list):
        status.text(f"获取 {i+1}/{total}: {name}")
        yf_ticker = code_to_yf(code)
        try:
            # Add jitter to avoid rate limiting
            time.sleep(random.uniform(0.1, 0.3))
            
            stock = yf.Ticker(yf_ticker)
            
            # Get stock info for fundamental data
            stock_info = stock.info
            trailing_eps = stock_info.get('trailingEps', 0)
            forward_eps = stock_info.get('forwardEps', 0)
            
            hist = stock.history(start=start_date_3m, end=end_date.strftime('%Y-%m-%d'))
            
            if not hist.empty and len(hist) >= 20:  # Need at least 20 days for 1m
                last = hist.iloc[-1]
                
                # Calculate returns for different periods
                # 1d return
                ret_1d = ((last['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close']) * 100 if len(hist) >= 2 else 0
                
                # 1w return (5 trading days)
                ret_1w = ((last['Close'] - hist.iloc[-6]['Close']) / hist.iloc[-6]['Close']) * 100 if len(hist) >= 6 else ret_1d
                
                # 1m return (21 trading days - approx 1 month)
                ret_1m = ((last['Close'] - hist.iloc[-22]['Close']) / hist.iloc[-22]['Close']) * 100 if len(hist) >= 22 else ret_1w
                
                # Calculate forward P/E metrics
                pe1 = last['Close'] / trailing_eps if trailing_eps and trailing_eps > 0 else None
                pe2 = last['Close'] / forward_eps if forward_eps and forward_eps > 0 else None
                
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
                    'trailing_eps': trailing_eps,
                    'forward_eps': forward_eps,
                    'pe1': pe1,  # P/E based on trailing EPS
                    'pe2': pe2,  # P/E based on forward EPS
                })
        except Exception as e:
            # If rate limited, skip and continue
            if "Too Many Requests" in str(e):
                st.warning(f"API限流，跳过 {name}")
            pass
        prog.progress((i+1)/total)
    
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
    
    # Avoid division by zero
    denominator = high_max - low_min
    denominator = denominator.replace(0, np.nan)
    
    df['RSV'] = 100 * ((df['Close'] - low_min) / denominator)
    df['RSV'] = df['RSV'].fillna(50)  # Fill NaN with neutral value
    
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
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(1)  # Fill NaN with 1 (neutral)
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
    try:
        df = calculate_macd(df)
        df = calculate_kdj(df)
        df = calculate_rsi(df)
        df = calculate_atr(df)
        df = calculate_bollinger_bands(df)
    except Exception as e:
        st.warning(f"技术指标计算警告: {e}")
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
# Get fundamental data from Yahoo Finance with better rate limiting
# ------------------------------------------------------------
@st.cache_data(ttl=7200)
def get_fundamental_data(yf_ticker):
    """Get fundamental data for a stock with rate limiting"""
    try:
        # Add delay to avoid rate limiting
        time.sleep(random.uniform(0.5, 1.0))
        
        stock = yf.Ticker(yf_ticker)
        info = stock.info
        
        # If we get rate limited, info might be empty
        if not info or len(info) < 10:
            return None, None
        
        # Key statistics with safe defaults
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
            'forward_eps': info.get('forwardEps', 0),
            'trailing_eps': info.get('trailingEps', 0),
        }
        
        return fundamentals, info
    except Exception as e:
        if "Too Many Requests" in str(e):
            st.warning("API限流，请稍后再试")
        return None, None

# ------------------------------------------------------------
# Calculate forward earnings growth and PEG ratios
# ------------------------------------------------------------
def calculate_forward_metrics(df):
    """Calculate EG1, EG2, PEG1, PEG2 based on earnings estimates"""
    df = df.copy()
    
    # Calculate PE ratios
    df['PE1'] = df['最新价'] / df['trailing_eps'].replace(0, np.nan)
    df['PE2'] = df['最新价'] / df['forward_eps'].replace(0, np.nan)
    
    # Calculate EG1 (growth from trailing to forward EPS)
    def calc_eg1(row):
        f0 = row['trailing_eps']
        f1 = row['forward_eps']
        if pd.isna(f0) or pd.isna(f1) or f0 == 0:
            return np.nan
        if f0 >= 0 >= f1:
            return -100
        elif f0 <= 0 < f1:
            return 100
        elif f0 < 0 and f1 <= 0:
            return -((f1 / f0) - 1) * 100
        else:
            return ((f1 / f0) - 1) * 100
    
    # Calculate EG2 (currently same as EG1 since we only have 2 periods)
    def calc_eg2(row):
        # For now, use same as EG1 since we don't have F2 estimates
        return calc_eg1(row)
    
    df['EG1'] = df.apply(calc_eg1, axis=1)
    df['EG2'] = df.apply(calc_eg2, axis=1)
    
    # Calculate PEG ratios
    df['PEG1'] = df['PE1'] / df['EG1'].abs().replace(0, np.nan)
    df['PEG2'] = df['PE2'] / df['EG2'].abs().replace(0, np.nan)
    
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
    st.markdown('<p class="main-header">📊 CSI 800 + CSI 1000 仪表盘</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">实时行情 + 技术分析 + 基本面深度挖掘 + 多周期板块轮动</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/investment-portfolio.png", width=100)
        st.title("控制面板")
        
        # Add a note about rate limiting
        st.info("⚠️ Yahoo Finance API 有访问限制，基本面数据可能需要较长时间加载")
        
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
    
    # Create a list of display options
    stock_options = df.apply(lambda x: f"{x['代码']} - {x['名称']}", axis=1).tolist()
    
    # Use columns for better layout
    col1, col2, col3, col4 = st.columns([2, 1.5, 1, 1])
    
    with col1:
        selected_option = st.selectbox("选择股票", options=stock_options, key="stock_selector")
        selected_code = selected_option.split(' - ')[0]
        stock_info = df[df['代码'] == selected_code].iloc[0]
    
    with col2:
        period = st.selectbox(
            "分析周期",
            options=["3mo", "6mo", "1y", "2y"],
            index=1,
            key="period_selector"
        )
    
    with col3:
        # Display current price
        price = stock_info['最新价']
        st.metric("当前价格", f"{price:.2f}")
    
    with col4:
        # Display daily movement
        change = stock_info['涨跌幅_1d']
        delta_color = "normal" if change > 0 else "inverse"
        st.metric("日涨跌幅", f"{change:+.2f}%", delta=f"{change:.2f}%", delta_color=delta_color)

    # --- Technical Analysis Section ---
    if selected_code:
        yf_ticker = code_to_yf(selected_code)
        
        with st.spinner("加载技术指标..."):
            # Fetch historical data
            stock = yf.Ticker(yf_ticker)
            hist = stock.history(period=period)
            
            if not hist.empty and len(hist) > 30:
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
                
                # Create K线图 (Candlestick chart) - Separate chart
                st.markdown("### 📊 K线图")
                fig_k = go.Figure()
                
                # Candlestick
                fig_k.add_trace(go.Candlestick(
                    x=hist_with_indicators.index,
                    open=hist_with_indicators['Open'],
                    high=hist_with_indicators['High'],
                    low=hist_with_indicators['Low'],
                    close=hist_with_indicators['Close'],
                    name='K线',
                    showlegend=False
                ))
                
                # Bollinger Bands
                fig_k.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['BB_Upper'],
                    line=dict(color='rgba(250, 128, 114, 0.5)', width=1),
                    name='上轨'
                ))
                
                fig_k.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['BB_Lower'],
                    line=dict(color='rgba(250, 128, 114, 0.5)', width=1),
                    name='下轨',
                    fill='tonexty',
                    fillcolor='rgba(250, 128, 114, 0.1)'
                ))
                
                fig_k.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['BB_Middle'],
                    line=dict(color='orange', width=1, dash='dash'),
                    name='中轨'
                ))
                
                fig_k.update_layout(
                    height=500,
                    title=f"{stock_info['名称']} ({selected_code}) - K线图",
                    hovermode='x unified',
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig_k, use_container_width=True)
                
                # Create成交量图 (Volume chart) - Separate chart
                st.markdown("### 📊 成交量图")
                fig_v = go.Figure()
                
                colors = ['red' if hist_with_indicators['Close'].iloc[i] >= hist_with_indicators['Open'].iloc[i] 
                         else 'green' for i in range(len(hist_with_indicators))]
                
                fig_v.add_trace(go.Bar(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['Volume'],
                    name='成交量',
                    marker_color=colors,
                    opacity=0.7
                ))
                
                fig_v.update_layout(
                    height=300,
                    title=f"{stock_info['名称']} ({selected_code}) - 成交量",
                    hovermode='x unified',
                    xaxis_rangeslider_visible=False,
                    yaxis_title="成交量"
                )
                
                st.plotly_chart(fig_v, use_container_width=True)
                
                # Create technical indicators chart (MACD, KDJ, RSI, ATR)
                st.markdown("### 📊 技术指标")
                fig_tech = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.25, 0.25, 0.25, 0.25],
                    subplot_titles=('MACD', 'KDJ', 'RSI', 'ATR & Bollinger Width')
                )
                
                # MACD
                fig_tech.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['MACD'],
                    line=dict(color='blue', width=2),
                    name='MACD'
                ), row=1, col=1)
                
                fig_tech.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['Signal'],
                    line=dict(color='red', width=2),
                    name='Signal'
                ), row=1, col=1)
                
                # MACD Histogram
                colors_macd = ['red' if x < 0 else 'green' for x in hist_with_indicators['MACD_Hist']]
                fig_tech.add_trace(go.Bar(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['MACD_Hist'],
                    name='MACD Hist',
                    marker_color=colors_macd,
                    opacity=0.5,
                    showlegend=False
                ), row=1, col=1)
                
                # KDJ
                fig_tech.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['K'],
                    line=dict(color='blue', width=2),
                    name='K'
                ), row=2, col=1)
                
                fig_tech.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['D'],
                    line=dict(color='red', width=2),
                    name='D'
                ), row=2, col=1)
                
                fig_tech.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['J'],
                    line=dict(color='green', width=2),
                    name='J'
                ), row=2, col=1)
                
                # RSI with levels
                fig_tech.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['RSI'],
                    line=dict(color='purple', width=2),
                    name='RSI'
                ), row=3, col=1)
                
                # RSI levels
                fig_tech.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
                fig_tech.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
                fig_tech.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
                
                # ATR and Bollinger Width
                fig_tech.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['ATR_pct'],
                    line=dict(color='brown', width=2),
                    name='ATR%'
                ), row=4, col=1)
                
                fig_tech.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['BB_Width'],
                    line=dict(color='orange', width=2, dash='dash'),
                    name='BB宽度%'
                ), row=4, col=1)
                
                fig_tech.update_layout(
                    height=800,
                    showlegend=True,
                    hovermode='x unified',
                    title=f"{stock_info['名称']} ({selected_code}) - 技术指标"
                )
                
                fig_tech.update_xaxes(rangeslider_visible=False)
                fig_tech.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
                
                st.plotly_chart(fig_tech, use_container_width=True)
                
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
            else:
                st.warning(f"股票 {stock_info['名称']} 历史数据不足，无法计算技术指标")

    # --- Fundamental Analysis Section ---
    st.markdown('<div class="section-header">📚 基本面深度分析</div>', unsafe_allow_html=True)
    
    if selected_code:
        with st.spinner("加载基本面数据 (可能需要10-20秒)..."):
            yf_ticker = code_to_yf(selected_code)
            fundamentals, info = get_fundamental_data(yf_ticker)
            
            if fundamentals and any(value != 0 for value in fundamentals.values()):
                # Key Metrics in columns
                st.markdown("### 关键指标")
                cols = st.columns(5)
                metrics = [
                    ('市值(亿)', f"{fundamentals['市值(亿)']:.0f}" if fundamentals['市值(亿)'] > 0 else 'N/A'),
                    ('PE(TTM)', f"{fundamentals['PE(TTM)']:.2f}" if fundamentals['PE(TTM)'] > 0 else 'N/A'),
                    ('PE(滚动)', f"{fundamentals['PE(滚动)']:.2f}" if fundamentals['PE(滚动)'] > 0 else 'N/A'),
                    ('PB', f"{fundamentals['PB']:.2f}" if fundamentals['PB'] > 0 else 'N/A'),
                    ('ROE(%)', f"{fundamentals['ROE(%)']:.1f}%" if fundamentals['ROE(%)'] > 0 else 'N/A'),
                ]
                
                for idx, (label, value) in enumerate(metrics):
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="fundamental-card">
                            <div style="color: #4B5563; font-size: 0.9rem;">{label}</div>
                            <div style="color: #111827; font-size: 1.5rem; font-weight: bold;">{value}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Valuation & Profitability
                st.markdown("### 💰 估值与盈利能力")
                col1, col2 = st.columns(2)
                
                with col1:
                    valuation_df = pd.DataFrame({
                        '指标': ['PEG', 'PS', '股息率(%)', 'Beta', 'EPS(滚动)'],
                        '数值': [
                            f"{fundamentals['PEG']:.2f}" if fundamentals['PEG'] > 0 else 'N/A',
                            f"{fundamentals['PS']:.2f}" if fundamentals['PS'] > 0 else 'N/A',
                            f"{fundamentals['股息率(%)']:.2f}%" if fundamentals['股息率(%)'] > 0 else 'N/A',
                            f"{fundamentals['Beta']:.2f}" if fundamentals['Beta'] > 0 else 'N/A',
                            f"{fundamentals['每股收益']:.2f}" if fundamentals['每股收益'] > 0 else 'N/A'
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
                
                # Forward Earnings Estimates
                st.markdown("### 📈 盈利预测")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="fundamental-card">
                        <div style="color: #4B5563;">当前EPS</div>
                        <div style="color: #111827; font-size: 1.3rem; font-weight: bold;">{fundamentals['每股收益']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="fundamental-card">
                        <div style="color: #4B5563;">预期EPS</div>
                        <div style="color: #111827; font-size: 1.3rem; font-weight: bold;">{fundamentals['forward_eps']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    if fundamentals['每股收益'] > 0 and fundamentals['forward_eps'] > 0:
                        growth = ((fundamentals['forward_eps'] / fundamentals['每股收益']) - 1) * 100
                        st.markdown(f"""
                        <div class="fundamental-card">
                            <div style="color: #4B5563;">预期增长</div>
                            <div style="color: {'#EF4444' if growth > 0 else '#10B981'}; font-size: 1.3rem; font-weight: bold;">{growth:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Price Levels
                st.markdown("### 📊 价格水平")
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
                            <div style="color: #4B5563;">当前价格: <span style="color: #111827; font-weight: bold;">{current:.2f}</span></div>
                            <div style="color: #4B5563;">52周高点: <span style="color: #111827;">{high_52w:.2f}</span> <span style="color: #EF4444;">(距离 {from_high:.1f}%)</span></div>
                            <div style="color: #4B5563;">52周低点: <span style="color: #111827;">{low_52w:.2f}</span> <span style="color: #10B981;">(距离 {from_low:.1f}%)</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Business Summary
                if info and 'longBusinessSummary' in info:
                    with st.expander("公司业务概览"):
                        st.write(info['longBusinessSummary'])
            else:
                st.warning("""
                ⚠️ 无法获取基本面数据 - Yahoo Finance API 访问限制
                
                可能的原因:
                - API 请求次数过多，请稍后再试
                - 该股票可能没有完整的基本面数据
                - 网络连接问题
                
                技术指标分析仍然可用，请继续使用上方的技术分析功能。
                """)
    else:
        st.info("请先选择一只股票进行基本面分析")

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
                for span in [20, 60, 120, 250]:
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
    
    # Check if the volume figures are different for each period
    vol_1d = df['成交量'].sum()
    vol_1w = df['成交量'].mean() * 5  # Approximate weekly volume
    vol_1m = df['成交量'].mean() * 21  # Approximate monthly volume
    
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
    
    # Add estimated volumes for different periods
    sector_stats['1d_成交额'] = sector_stats['成交额(亿)']
    sector_stats['1w_成交额'] = (sector_stats['成交额(亿)'] * 5).round(2)
    sector_stats['1m_成交额'] = (sector_stats['成交额(亿)'] * 21).round(2)
    
    period_tabs = st.tabs(["1日表现", "1周表现", "1月表现"])
    
    for idx, (period, ret_col, vol_col) in enumerate([
        ("1日", "1d_平均涨跌幅", "1d_成交额"),
        ("1周", "1w_平均涨跌幅", "1w_成交额"),
        ("1月", "1m_平均涨跌幅", "1m_成交额")
    ]):
        with period_tabs[idx]:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig = px.scatter(
                    sector_stats,
                    x=ret_col,
                    y=vol_col,
                    size='成分股数',
                    color=ret_col,
                    text='板块',
                    title=f'板块轮动气泡图 ({period}表现)',
                    color_continuous_scale='RdYlGn',
                    size_max=50,
                    labels={ret_col: f'平均涨跌幅 (%)', vol_col: f'成交额 (亿)'}
                )
                fig.update_traces(textposition='top center')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                display_df = sector_stats[[
                    '板块', ret_col, f'{ret_col.split("_")[0]}_波动率', '成分股数', vol_col
                ]].copy()
                display_df.columns = ['板块', '平均涨跌幅(%)', '波动率', '成分股数', '成交额(亿)']
                display_df = display_df.sort_values('平均涨跌幅(%)', ascending=False)
                
                display_df_display = display_df.copy()
                display_df_display['平均涨跌幅(%)'] = display_df_display['平均涨跌幅(%)'].apply(lambda x: f"{x:.2f}%")
                display_df_display['波动率'] = display_df_display['波动率'].apply(lambda x: f"{x:.2f}%")
                display_df_display['成交额(亿)'] = display_df_display['成交额(亿)'].apply(lambda x: f"{x:.0f}")
                
                st.dataframe(
                    display_df_display,
                    use_container_width=True,
                    hide_index=True,
                    height=500
                )

    # --- Sector Momentum Comparison ---
    st.markdown('<div class="section-header">📊 板块动量对比</div>', unsafe_allow_html=True)
    
    # Show ALL sectors, not just top 15
    momentum_df = sector_stats[['板块', '1d_平均涨跌幅', '1w_平均涨跌幅', '1m_平均涨跌幅']].copy()
    momentum_df = momentum_df.sort_values('1m_平均涨跌幅', ascending=False)  # No head(15) - show all
    
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
    
    # Use a more colorful palette with better contrast
    fig = px.bar(
        momentum_melted,
        x='板块',
        y='涨跌幅',
        color='周期',
        barmode='group',
        title='全板块多周期动量对比',
        labels={'涨跌幅': '涨跌幅 (%)', '板块': ''},
        color_discrete_sequence=['#1E88E5', '#FFC107', '#DC143C']  # Blue, Amber, Crimson - better contrast
    )
    fig.update_layout(
        height=max(400, len(momentum_df) * 20),  # Dynamic height based on number of sectors
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Advanced Stock Selection with PEG and Growth Criteria ---
    st.markdown('<div class="section-header">🎯 基本面精选股 (PEG & 增长筛选)</div>', unsafe_allow_html=True)

    # Adjusted market cap options based on actual data distribution
    market_cap_options = {
        "全部": (0, float('inf')),
        "微盘股(<10亿)": (0, 10),        # Below median
        "小盘股(10-50亿)": (10, 50),      # Median to 2x median
        "中盘股(50-150亿)": (50, 150),    # Upper quartile range
        "大盘股(>150亿)": (150, float('inf'))  # Top end
    }
    
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_mcap_range = st.selectbox("选择市值范围", list(market_cap_options.keys()), index=2)  # Default to 10-50亿
    
    # Calculate forward metrics (simplified - only use EG1)
    def calculate_forward_metrics_simple(df):
        """Calculate EG1 and PEG1 based on earnings estimates"""
        df = df.copy()
        
        # Estimate market cap (in billions)
        df['市值(亿)'] = df['最新价'] * df['成交量'] * 2 / 1e8  # Rough estimate using average of 2 days volume
        
        # Calculate PE ratios
        df['PE0'] = df['最新价'] / df['trailing_eps'].replace(0, np.nan)  # Current P/E (was PE1)
        df['PE1'] = df['最新价'] / df['forward_eps'].replace(0, np.nan)   # Forward P/E (was PE2)
        
        # Calculate EG1 (growth from trailing to forward EPS)
        def calc_eg1(row):
            f0 = row['trailing_eps']
            f1 = row['forward_eps']
            if pd.isna(f0) or pd.isna(f1) or f0 == 0:
                return np.nan
            if f0 >= 0 >= f1:
                return -100
            elif f0 <= 0 < f1:
                return 100
            elif f0 < 0 and f1 <= 0:
                return -((f1 / f0) - 1) * 100
            else:
                return ((f1 / f0) - 1) * 100
        
        df['EG1'] = df.apply(calc_eg1, axis=1)
        
        # Calculate PEG ratio (using forward P/E)
        df['PEG1'] = df['PE1'] / df['EG1'].abs().replace(0, np.nan)
        
        return df
    
    # Apply calculations
    df_with_metrics = calculate_forward_metrics_simple(df)
    
    # Apply market cap filter
    min_mcap, max_mcap = market_cap_options[selected_mcap_range]
    if max_mcap != float('inf'):
        df_filtered = df_with_metrics[(df_with_metrics['市值(亿)'] >= min_mcap) & (df_with_metrics['市值(亿)'] <= max_mcap)]
    else:
        df_filtered = df_with_metrics[df_with_metrics['市值(亿)'] >= min_mcap]
    
    # Drop rows with missing data
    df_clean = df_filtered.dropna(subset=['最新价', 'trailing_eps', 'forward_eps', 'PE0', 'PE1', 'EG1', 'PEG1'])
    
    if not df_clean.empty:
        # Display market cap range info
        mcap_info = f"当前筛选: {selected_mcap_range}"
        if min_mcap > 0:
            mcap_info += f" ({min_mcap}-{max_mcap if max_mcap != float('inf') else '∞'}亿)"
        st.caption(mcap_info)
        
        selected_universe = pd.DataFrame()
        sectors = df_clean['板块'].unique()
        
        # Apply filtering criteria per sector
        for curr_sector in sectors:
            curr_universe = df_clean[df_clean['板块'] == curr_sector].copy()
            
            if curr_universe.empty:
                continue
                
            # Calculate sector medians
            sector_pe0 = curr_universe['PE0'].median()
            sector_pe1 = curr_universe['PE1'].median()
            sector_eg1 = curr_universe['EG1'].median()
            
            # PE conditions
            pe_condition = curr_universe['PE0'] > curr_universe['PE1']  # Forward P/E lower than current P/E
            pe_condition &= curr_universe['PE0'] > sector_pe0  # Current P/E above sector median
            pe_condition &= curr_universe['PE1'] > sector_pe1  # Forward P/E above sector median
            
            # Growth condition - positive growth and above sector median
            eg_condition = curr_universe['EG1'] > 0  # Positive expected growth
            eg_condition &= curr_universe['EG1'] > sector_eg1  # Growth above sector median
            
            # PEG condition
            peg_condition = curr_universe['PEG1'] < 1  # PEG < 1 (undervalued relative to growth)
            peg_condition &= curr_universe['PEG1'] > 0  # Positive PEG
            
            # Combine all conditions
            curr_selected_universe = curr_universe[pe_condition & eg_condition & peg_condition]
            
            if not curr_selected_universe.empty:
                selected_universe = pd.concat([selected_universe, curr_selected_universe])
        
        if not selected_universe.empty:
            # Display results
            display_cols = ['代码', '名称', '板块', '最新价', '市值(亿)', '涨跌幅_1d', 'PE0', 'PE1', 'EG1', 'PEG1']
            display_df = selected_universe[display_cols].copy()
            
            # Format columns
            display_df['涨跌幅_1d'] = display_df['涨跌幅_1d'].apply(lambda x: f"{x:.2f}%")
            display_df['市值(亿)'] = display_df['市值(亿)'].apply(lambda x: f"{x:.0f}")
            display_df['PE0'] = display_df['PE0'].apply(lambda x: f"{x:.2f}")
            display_df['PE1'] = display_df['PE1'].apply(lambda x: f"{x:.2f}")
            display_df['EG1'] = display_df['EG1'].apply(lambda x: f"{x:.1f}%")
            display_df['PEG1'] = display_df['PEG1'].apply(lambda x: f"{x:.2f}")
            
            st.success(f"找到 {len(selected_universe)} 只符合PEG<1且增长为正的股票 (市值范围: {selected_mcap_range})")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Summary statistics
            st.markdown("### 📊 筛选条件说明")
            st.info("""
            **筛选逻辑:**
            - **PE0 > PE1** (远期PE低于当前PE) - 估值改善
            - **PE0 > 行业平均, PE1 > 行业平均** - 相对估值合理
            - **EG1 > 0** (预期增长为正)
            - **EG1 > 行业平均** - 增长高于同行
            - **0 < PEG1 < 1** (估值合理且增长有吸引力)
            
            **PE0**: 当前市盈率 (基于过去12个月盈利)  
            **PE1**: 远期市盈率 (基于未来12个月预期盈利)  
            **EG1**: 预期盈利增长率  
            **PEG1**: 市盈率相对盈利增长比率
            """)
        else:
            st.warning(f"在{selected_mcap_range}范围内，没有股票满足所有筛选条件，请尝试放宽条件或选择其他市值范围")
    else:
        st.warning(f"在{selected_mcap_range}范围内，EPS数据不足，无法进行PEG筛选")
    
    # Add a note about market cap distribution
    if not df_with_metrics.empty:
        mcap_stats = df_with_metrics['市值(亿)'].describe()
        st.caption(f"📊 全市场市值分布: 平均 {mcap_stats['mean']:.0f}亿 | 中位数 {mcap_stats['50%']:.0f}亿 | 最小 {mcap_stats['min']:.0f}亿 | 最大 {mcap_stats['max']:.0f}亿")

    # --- Sector Stock Selector (Original) ---
    st.markdown('<div class="section-header">🔍 板块内选股 (动量+成交额)</div>', unsafe_allow_html=True)
    selected_sector = st.selectbox("选择板块", ['全部'] + sorted(df['板块'].unique()), key='sector_selector')
    if selected_sector != '全部':
        sector_df = df[df['板块'] == selected_sector].copy()
    else:
        sector_df = df.copy()

    # Avoid division by zero in scoring
    if sector_df['涨跌幅_1d'].std() > 0:
        sector_df['动量分'] = (sector_df['涨跌幅_1d'] - sector_df['涨跌幅_1d'].mean()) / sector_df['涨跌幅_1d'].std()
    else:
        sector_df['动量分'] = 0
        
    if sector_df['成交额(亿)'].std() > 0:
        sector_df['成交额分'] = (sector_df['成交额(亿)'] - sector_df['成交额(亿)'].mean()) / sector_df['成交额(亿)'].std()
    else:
        sector_df['成交额分'] = 0
        
    sector_df['综合分'] = (sector_df['动量分']*0.6 + sector_df['成交额分']*0.4).round(2)

    top_stocks = sector_df.nlargest(15, '综合分')[['代码','名称','板块','最新价','涨跌幅_1d','涨跌幅_1w','涨跌幅_1m','成交额(亿)','综合分']]
    top_stocks['涨跌幅_1d'] = top_stocks['涨跌幅_1d'].apply(lambda x: f"{x:.2f}%")
    top_stocks['涨跌幅_1w'] = top_stocks['涨跌幅_1w'].apply(lambda x: f"{x:.2f}%")
    top_stocks['涨跌幅_1m'] = top_stocks['涨跌幅_1m'].apply(lambda x: f"{x:.2f}%")
    st.dataframe(top_stocks, use_container_width=True, hide_index=True)

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

    top_sectors_1d = sector_perf_1d.head(3).index.tolist() if not sector_perf_1d.empty else []
    top_sectors_1w = sector_perf_1w.head(3).index.tolist() if not sector_perf_1w.empty else []
    top_sectors_1m = sector_perf_1m.head(3).index.tolist() if not sector_perf_1m.empty else []
    
    st.markdown(f"""
    <div class="strategy-box">
        <h3>当前市场状态: <span style="color:{color};">{regime} ({trend_strength})</span></h3>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem;">
            <div><span class="metric-label">建议仓位</span><div class="metric-value">{pos}</div></div>
            <div><span class="metric-label">风险水平</span><div class="metric-value">{'高' if vola_1d>2 else '中' if vola_1d>1 else '低'}</div></div>
            <div><span class="metric-label">操作策略</span><div class="metric-value">{action}</div></div>
        </div>
        <div style="margin-top:1.5rem; border-top:1px solid #ddd; padding-top:1rem;">
            <p><strong>短期强势板块(1日):</strong> {', '.join(top_sectors_1d[:3]) if top_sectors_1d else 'N/A'}</p>
            <p><strong>中期强势板块(1周):</strong> {', '.join(top_sectors_1w[:3]) if top_sectors_1w else 'N/A'}</p>
            <p><strong>长期强势板块(1月):</strong> {', '.join(top_sectors_1m[:3]) if top_sectors_1m else 'N/A'}</p>
            <p><strong>选股参考:</strong> 板块内综合分 > 0 | 成交额 > 板块平均 | 多周期动量向上</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:#666; font-size:0.8rem;">
        数据来源: Yahoo Finance | 成分股列表: universe.csv | 对冲基金级分析系统 v3.0<br>
        技术指标: MACD(12,26,9) | KDJ(9,3,3) | RSI(14) | ATR(14) | Bollinger Bands(20,2)<br>
        基本面: 当前EPS, 预期EPS, PEG, 增长加速筛选<br>
        更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
