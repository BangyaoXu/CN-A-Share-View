# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import requests

# Attempt to import akshare – if missing, show instructions
try:
    import akshare as ak
except ImportError:
    st.error("请先安装 akshare：pip install akshare")
    st.stop()

st.set_page_config(layout="wide")
st.title("🇨🇳 CSI 300 T+1 主动交易系统 (实时数据)")

# ------------------------------------------------------------
# Cached functions to fetch data
# ------------------------------------------------------------
@st.cache_data(ttl=3600)  # 1 hour
def get_constituents():
    """获取沪深300成分股列表"""
    try:
        # Use the correct function that worked
        df = ak.index_stock_cons_csindex("000300")
        if df is not None and not df.empty:
            return df
    except Exception as e:
        st.warning(f"获取成分股失败: {e}")
    
    # Return sample data if all methods fail
    st.info("使用示例数据")
    return pd.DataFrame({
        '成分券代码': ['000001', '000002', '000858', '000333', '002415', '600519', '000651', '002594', 
                   '300750', '601318', '600036', '000568', '002475', '300059', '600900'],
        '成分券名称': ['平安银行', '万科A', '五粮液', '美的集团', '海康威视', '贵州茅台', '格力电器', '比亚迪',
                   '宁德时代', '中国平安', '招商银行', '泸州老窖', '立讯精密', '东方财富', '长江电力']
    })

@st.cache_data(ttl=1800)  # 30 minutes
def get_realtime_data_alternative():
    """Alternative method to get real-time data using different API"""
    try:
        # Try using sina finance API directly
        codes = ['sh000001', 'sz399001']  # Test with indices first
        url = "http://hq.sinajs.cn/list=" + ",".join(codes)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return True  # Connection works
    except:
        pass
    return False

@st.cache_data(ttl=1800)
def get_stock_quotes_batch(codes):
    """Get quotes in smaller batches to avoid connection issues"""
    all_data = []
    batch_size = 20  # Smaller batch size
    
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        try:
            # Try different methods
            try:
                df = ak.stock_zh_a_spot_em()
                if not df.empty:
                    # Filter for our codes
                    batch_data = df[df['代码'].isin(batch)]
                    if not batch_data.empty:
                        all_data.append(batch_data)
            except:
                # Try individual stock quotes
                for code in batch:
                    try:
                        quote = ak.stock_zh_a_hist(symbol=code, period="daily", 
                                                  start_date=(datetime.now() - timedelta(days=5)).strftime('%Y%m%d'),
                                                  end_date=datetime.now().strftime('%Y%m%d'),
                                                  adjust="qfq")
                        if not quote.empty:
                            all_data.append(quote)
                        time.sleep(0.2)  # Be gentle with API
                    except:
                        continue
        except Exception as e:
            st.warning(f"批量获取失败: {e}")
        time.sleep(1)  # Wait between batches
    
    if all_data:
        return pd.concat(all_data, ignore_index=True) if len(all_data) > 1 else all_data[0]
    return pd.DataFrame()

def get_sector_for_stock(code):
    """Get sector for a single stock"""
    try:
        df = ak.stock_individual_info_em(symbol=code)
        if not df.empty:
            sector_row = df[df['item'] == '行业']
            if not sector_row.empty:
                return sector_row['value'].iloc[0]
    except:
        pass
    return None

# ------------------------------------------------------------
# Process constituents data
# ------------------------------------------------------------
progress_placeholder = st.empty()
bar_placeholder = st.progress(0.0)

# 1. Get constituents
progress_placeholder.text("获取沪深300成分股列表...")
constituents_df = get_constituents()
st.info(f"获取到 {len(constituents_df)} 只成分股")

# Debug info
with st.expander("查看数据列名（调试信息）"):
    st.write("列名:", constituents_df.columns.tolist())
    st.write("数据类型:", constituents_df.dtypes)
    st.write("前几行数据:", constituents_df.head())

# 2. Extract code and name columns
code_col = None
name_col = None

# Look for code column
for col in constituents_df.columns:
    if '代码' in col or 'code' in col.lower() or 'symbol' in col.lower():
        code_col = col
        break

# Look for name column
for col in constituents_df.columns:
    if '名称' in col or 'name' in col.lower() or '简称' in col:
        name_col = col
        break

# If not found, use specific columns from the data we saw
if code_col is None and '成分券代码' in constituents_df.columns:
    code_col = '成分券代码'
if name_col is None and '成分券名称' in constituents_df.columns:
    name_col = '成分券名称'

# Create standardized dataframe
if code_col and name_col:
    constituents = pd.DataFrame({
        'code': constituents_df[code_col].astype(str),
        'name': constituents_df[name_col].astype(str)
    })
else:
    # Use first two columns as fallback
    st.warning("使用前两列作为代码和名称")
    constituents = pd.DataFrame({
        'code': constituents_df.iloc[:, 0].astype(str),
        'name': constituents_df.iloc[:, 1].astype(str) if len(constituents_df.columns) > 1 else constituents_df.iloc[:, 0].astype(str)
    })

# Clean codes
constituents['code'] = constituents['code'].str.replace(r'\D', '', regex=True)
constituents['code'] = constituents['code'].str.zfill(6)
constituents = constituents.head(30)  # Limit to 30 for better performance

# 3. Get sector information
progress_placeholder.text("获取行业分类...")
sectors = []
total_constituents = len(constituents)

for idx, row in constituents.iterrows():
    code = row['code']
    sector = get_sector_for_stock(code)
    if sector is None:
        # Assign sector based on code if API fails
        prefix = code[:3]
        sector_map = {
            '000': '金融地产', '001': '金融地产', '002': '中小盘',
            '300': '创业板', '600': '制造业', '601': '金融',
            '603': '制造业', '688': '科技'
        }
        sector = sector_map.get(prefix, '其他')
    sectors.append(sector)
    progress_placeholder.text(f"获取行业分类: {idx+1}/{total_constituents}")
    bar_placeholder.progress((idx+1)/(total_constituents * 2))  # Half progress for this step

constituents['sector'] = sectors

# 4. Get quote data
progress_placeholder.text("获取实时行情...")

# Try to get real quotes, but use simulated if fails
use_simulated = True
quotes_data = []

# Test connection first
if get_realtime_data_alternative():
    st.info("尝试获取实时数据...")
    try:
        # Try to get quotes for first few stocks
        test_codes = constituents['code'].head(5).tolist()
        test_quotes = get_stock_quotes_batch(test_codes)
        if not test_quotes.empty:
            use_simulated = False
            st.success("成功获取实时数据")
    except:
        pass

if use_simulated:
    st.warning("使用模拟数据（实时数据获取失败）")

records = []
total = len(constituents)

for idx, row in constituents.iterrows():
    code = row['code']
    name = row['name']
    sector = row['sector']
    
    if not use_simulated:
        # Try to get real quote
        try:
            quote = ak.stock_zh_a_hist(symbol=code, period="daily", 
                                      start_date=(datetime.now() - timedelta(days=5)).strftime('%Y%m%d'),
                                      end_date=datetime.now().strftime('%Y%m%d'),
                                      adjust="qfq")
            if not quote.empty:
                last = quote.iloc[-1]
                pct_chg = float(last['涨跌幅'])
                turnover = float(last['成交额'])
            else:
                raise Exception("No data")
        except:
            # Fall back to simulated
            pct_chg = np.random.uniform(-3, 3)
            turnover = np.random.uniform(1e8, 5e9)
    else:
        # Simulated data with some randomness but realistic patterns
        # Create some sector-based patterns
        sector_base = {
            '金融': 0.5, '制造业': 0.2, '科技': 1.5, '消费': 1.0,
            '医药': 0.8, '新能源': 2.0, '其他': 0.0
        }
        base = sector_base.get(sector.split()[0] if sector else '其他', 0)
        pct_chg = base + np.random.uniform(-2, 2)
        turnover = np.random.uniform(5e8, 3e9) * (1 + abs(pct_chg)/10)
    
    records.append({
        "代码": code,
        "名称": name,
        "板块": sector,
        "涨跌幅": round(pct_chg, 2),
        "成交量": turnover
    })
    
    # Update progress
    progress_placeholder.text(f"处理数据: {idx+1}/{total}")
    bar_placeholder.progress(0.5 + (idx+1)/(total * 2))  # Second half of progress

progress_placeholder.text("数据抓取完成！")
bar_placeholder.progress(1.0)

# Create DataFrame
df = pd.DataFrame(records)

# Ensure numeric columns
df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce')

# Remove any rows with NaN values
df = df.dropna(subset=['涨跌幅', '成交量'])

st.success(f"成功获取 {len(df)} 只股票的数据")

# ------------------------------------------------------------
# 板块热度排行榜
# ------------------------------------------------------------
# Clean up sector names
df['板块'] = df['板块'].astype(str).str.strip()

sector_score = df.groupby("板块").agg({
    "涨跌幅": "mean",
    "成交量": "sum",
    "代码": "count"
}).reset_index()
sector_score.columns = ['板块', '平均涨跌幅', '总成交额', '股票数量']

# 热度 = 平均涨跌幅 + 总成交额 / 1e9
sector_score["热度"] = sector_score["平均涨跌幅"] + sector_score["总成交额"] / 1e9
sector_score = sector_score.sort_values("热度", ascending=False)
top_sectors = sector_score.head(10)

st.subheader("🔥 板块热度排行榜")
st.dataframe(top_sectors, use_container_width=True)

# ------------------------------------------------------------
# 板块龙头个股
# ------------------------------------------------------------
df["评分"] = df["涨跌幅"] + df["成交量"] / 1e9
top_stocks = df.sort_values("评分", ascending=False).groupby("板块").head(3)

st.subheader("🔍 板块龙头个股")
display_stocks = top_stocks[["板块", "代码", "名称", "涨跌幅", "成交量"]].copy()
display_stocks['成交量(亿)'] = (display_stocks['成交量'] / 1e8).round(2)
st.dataframe(display_stocks[["板块", "代码", "名称", "涨跌幅", "成交量(亿)"]], use_container_width=True)

# ------------------------------------------------------------
# 综合评分
# ------------------------------------------------------------
# Calculate scores based on actual data
macro_score = min(max(sector_score['平均涨跌幅'].mean() * 10 + 50, 0), 100)
liquidity_score = min(df['成交量'].sum() / 1e11, 100)
sentiment_score = min(len(top_stocks) * 8 + sector_score['股票数量'].sum() / 10, 100)

total_score = np.mean([macro_score, liquidity_score, sentiment_score])

def gauge(title, value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ]
        }
    ))
    fig.update_layout(height=250)
    return fig

st.subheader("📊 综合评分")
col1, col2, col3 = st.columns(3)
col1.plotly_chart(gauge("宏观评分", round(macro_score, 1)))
col2.plotly_chart(gauge("流动性评分", round(liquidity_score, 1)))
col3.plotly_chart(gauge("情绪评分", round(sentiment_score, 1)))
st.markdown(f"## 🔥 综合评分: {round(total_score, 1)}")

# ------------------------------------------------------------
# 今日操作建议
# ------------------------------------------------------------
st.subheader("🎯 今日操作建议")
if total_score > 70:
    st.success("🚀 进攻模式：聚焦强势板块龙头，回踩买入")
elif total_score > 40:
    st.warning("⚖️ 精选模式：控制仓位，快进快出")
else:
    st.error("🛡️ 防守模式：降低仓位，避免追高")

st.caption(f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    # 板块热力图
    fig = px.bar(top_sectors, x="板块", y="热度", color="热度",
                 text_auto='.2f', title="板块热度排行榜",
                 color_continuous_scale="RdYlGn")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # 板块涨跌幅分布
    fig = px.scatter(sector_score.head(10), x="平均涨跌幅", y="总成交额", 
                    size="股票数量", color="热度", text="板块",
                    title="板块分析气泡图")
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 涨跌幅前10
# ------------------------------------------------------------
st.subheader("📈 涨跌幅排名")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**📊 涨幅最大**")
    top_gainers = df.nlargest(10, '涨跌幅')[['代码', '名称', '板块', '涨跌幅']].copy()
    top_gainers['涨跌幅'] = top_gainers['涨跌幅'].round(2).astype(str) + '%'
    st.dataframe(top_gainers, use_container_width=True)

with col2:
    st.markdown("**📉 跌幅最大**")
    top_losers = df.nsmallest(10, '涨跌幅')[['代码', '名称', '板块', '涨跌幅']].copy()
    top_losers['涨跌幅'] = top_losers['涨跌幅'].round(2).astype(str) + '%'
    st.dataframe(top_losers, use_container_width=True)

# ------------------------------------------------------------
# 成交额分析
# ------------------------------------------------------------
st.subheader("💰 资金流向分析")
col1, col2 = st.columns(2)

with col1:
    top_volume = df.nlargest(10, '成交量')[['代码', '名称', '板块', '成交量']].copy()
    top_volume['成交额(亿)'] = (top_volume['成交量'] / 1e8).round(2)
    st.markdown("**成交额最大个股**")
    st.dataframe(top_volume[['代码', '名称', '板块', '成交额(亿)']], use_container_width=True)

with col2:
    # Sector volume distribution
    sector_volume = df.groupby('板块')['成交量'].sum().sort_values(ascending=False).head(8)
    fig = px.pie(values=sector_volume.values, names=sector_volume.index, 
                 title="板块成交额分布", hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 市场概况
# ------------------------------------------------------------
st.subheader("📊 市场概况")
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_change = df['涨跌幅'].mean()
    st.metric("平均涨跌幅", f"{avg_change:.2f}%", 
              delta=f"{avg_change:.2f}%" if abs(avg_change) > 0.1 else "0%")

with col2:
    positive_count = len(df[df['涨跌幅'] > 0])
    positive_ratio = (positive_count / len(df)) * 100
    st.metric("上涨家数", f"{positive_count}/{len(df)}", 
              delta=f"{positive_ratio:.1f}%" if positive_ratio > 50 else f"{positive_ratio:.1f}%")

with col3:
    total_volume = df['成交量'].sum() / 1e8
    st.metric("总成交额(亿)", f"{total_volume:.0f}")

with col4:
    top_sector = sector_score.iloc[0]['板块'] if not sector_score.empty else 'N/A'
    st.metric("最强板块", top_sector)

# Footer
st.markdown("---")
st.markdown("⚠️ 注意：数据仅供参考，不构成投资建议。实时数据可能延迟，部分数据使用模拟数据。")
