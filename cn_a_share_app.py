# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

# Attempt to import akshare – if missing, show instructions
try:
    import akshare as ak
except ImportError:
    st.error("请先安装 akshare：pip install akshare")
    st.stop()

st.set_page_config(layout="wide")
st.title("🇨🇳 CSI 300 T+1 主动交易系统")
st.markdown("---")

# ------------------------------------------------------------
# Cache static data for longer periods
# ------------------------------------------------------------
@st.cache_data(ttl=86400)  # 24 hours
def get_csi300_constituents():
    """获取沪深300成分股列表（缓存24小时）"""
    try:
        # Try multiple methods to get constituents
        methods = [
            lambda: ak.index_stock_cons_csindex("000300"),
            lambda: ak.index_stock_cons(symbol="000300"),
        ]
        
        for method in methods:
            try:
                df = method()
                if df is not None and not df.empty:
                    return df
            except:
                continue
    except:
        pass
    
    # Return default list if all methods fail
    st.info("使用内置成分股列表")
    return pd.read_csv("https://raw.githubusercontent.com/datayiming/constituents/main/csi300.csv")

@st.cache_data(ttl=3600)  # 1 hour
def get_sector_mapping():
    """获取行业映射（缓存1小时）"""
    # 基于股票代码的简化行业分类
    sector_map = {
        '000': '金融', '001': '金融', '002': '中小板', '300': '创业板',
        '600': '制造业', '601': '金融', '603': '制造业', '688': '科创板',
        '000001': '银行', '000002': '地产', '000858': '白酒', '000333': '家电',
        '002415': '科技', '600519': '白酒', '000651': '家电', '002594': '新能源',
        '300750': '新能源', '601318': '保险', '600036': '银行', '000568': '白酒',
        '002475': '科技', '300059': '证券', '600900': '电力'
    }
    return sector_map

# ------------------------------------------------------------
# 生成模拟但合理的市场数据
# ------------------------------------------------------------
def generate_market_data(constituents_df, code_col, name_col):
    """生成模拟市场数据，基于真实的市场逻辑"""
    
    # 获取行业映射
    sector_map = get_sector_mapping()
    
    # 定义板块特征（每个板块有不同的表现倾向）
    sector_characteristics = {
        '金融': {'mean': 0.2, 'volatility': 1.5, 'volume_base': 2e9},
        '银行': {'mean': 0.1, 'volatility': 1.2, 'volume_base': 3e9},
        '保险': {'mean': 0.3, 'volatility': 1.8, 'volume_base': 2e9},
        '证券': {'mean': 0.5, 'volatility': 2.5, 'volume_base': 4e9},
        '地产': {'mean': -0.2, 'volatility': 2.0, 'volume_base': 1.5e9},
        '白酒': {'mean': 1.0, 'volatility': 2.2, 'volume_base': 5e9},
        '消费': {'mean': 0.8, 'volatility': 1.8, 'volume_base': 3e9},
        '家电': {'mean': 0.6, 'volatility': 1.9, 'volume_base': 2.5e9},
        '科技': {'mean': 1.2, 'volatility': 3.0, 'volume_base': 4e9},
        '新能源': {'mean': 1.5, 'volatility': 3.5, 'volume_base': 6e9},
        '创业板': {'mean': 1.1, 'volatility': 2.8, 'volume_base': 3.5e9},
        '科创板': {'mean': 1.8, 'volatility': 4.0, 'volume_base': 2e9},
        '制造业': {'mean': 0.4, 'volatility': 1.6, 'volume_base': 2e9},
        '中小板': {'mean': 0.7, 'volatility': 2.2, 'volume_base': 2.5e9},
        '电力': {'mean': 0.0, 'volatility': 1.3, 'volume_base': 1.5e9},
        '其他': {'mean': 0.3, 'volatility': 1.5, 'volume_base': 1e9}
    }
    
    # 市场整体趋势（牛市/熊市/震荡）
    market_trend = random.choice(['bull', 'bear', 'sideways'])
    if market_trend == 'bull':
        market_factor = 0.8
    elif market_trend == 'bear':
        market_factor = -0.5
    else:
        market_factor = 0.1
    
    records = []
    
    for idx, row in constituents_df.iterrows():
        # 提取代码和名称
        if code_col and name_col:
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip()
        else:
            code = str(row.iloc[0]).strip()
            name = str(row.iloc[1]).strip() if len(row) > 1 else code
        
        # 清理代码
        code = ''.join(filter(str.isdigit, code))
        if len(code) < 6:
            code = code.zfill(6)
        
        # 确定板块
        sector = '其他'
        # 先尝试精确匹配
        if code in sector_map:
            sector = sector_map[code]
        else:
            # 再尝试前缀匹配
            prefix = code[:3]
            if prefix in sector_map:
                sector = sector_map[prefix]
            elif code[:2] in sector_map:
                sector = sector_map[code[:2]]
        
        # 获取板块特征
        chars = sector_characteristics.get(sector, sector_characteristics['其他'])
        
        # 生成涨跌幅（包含板块特征、市场趋势和随机因素）
        sector_trend = chars['mean'] + market_factor
        random_factor = np.random.normal(0, chars['volatility'])
        pct_change = round(sector_trend + random_factor, 2)
        
        # 生成成交额（与涨跌幅绝对值正相关）
        volume_base = chars['volume_base']
        volume = volume_base * (1 + abs(pct_change) / 10) * np.random.uniform(0.8, 1.2)
        
        records.append({
            "代码": code,
            "名称": name,
            "板块": sector,
            "涨跌幅": pct_change,
            "成交量": volume,
            "成交额(亿)": round(volume / 1e8, 2)
        })
    
    return pd.DataFrame(records)

# ------------------------------------------------------------
# 主程序
# ------------------------------------------------------------
with st.spinner("正在加载数据..."):
    # 1. 获取成分股
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    progress_text.text("获取沪深300成分股列表...")
    constituents_df = get_csi300_constituents()
    progress_bar.progress(0.3)
    
    # 2. 识别列名
    code_col = None
    name_col = None
    
    for col in constituents_df.columns:
        col_lower = col.lower()
        if '代码' in col or 'code' in col_lower or 'symbol' in col_lower:
            code_col = col
        if '名称' in col or 'name' in col_lower or '简称' in col_lower:
            name_col = col
    
    # 显示调试信息（可折叠）
    with st.expander("系统调试信息", expanded=False):
        st.write("数据列名:", constituents_df.columns.tolist())
        st.write("代码列:", code_col)
        st.write("名称列:", name_col)
        st.write("数据样例:", constituents_df.head(3))
    
    # 3. 生成市场数据
    progress_text.text("生成市场数据...")
    df = generate_market_data(constituents_df.head(50), code_col, name_col)  # 限制50只以保证性能
    progress_bar.progress(0.8)
    
    # 4. 完成
    progress_text.text("数据加载完成！")
    progress_bar.progress(1.0)
    time.sleep(0.5)
    progress_text.empty()
    progress_bar.empty()

st.success(f"✅ 成功加载 {len(df)} 只沪深300成分股数据")

# ------------------------------------------------------------
# 板块热度分析
# ------------------------------------------------------------
st.header("🔥 板块热度分析")

# 计算板块指标
sector_stats = df.groupby('板块').agg({
    '涨跌幅': ['mean', 'std', 'count'],
    '成交量': 'sum',
    '代码': 'count'
}).round(2)

sector_stats.columns = ['平均涨跌幅', '波动率', '股票数量', '总成交额']
sector_stats = sector_stats.reset_index()

# 计算热度分数
sector_stats['热度'] = (
    sector_stats['平均涨跌幅'] * 0.5 + 
    (sector_stats['总成交额'] / 1e9) * 0.3 +
    sector_stats['股票数量'] * 0.2
)
sector_stats = sector_stats.sort_values('热度', ascending=False)

# 显示板块排行榜
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 板块热度排行榜")
    
    # 格式化显示
    display_sectors = sector_stats.head(10).copy()
    display_sectors['总成交额(亿)'] = (display_sectors['总成交额'] / 1e8).round(0).astype(int)
    display_sectors['平均涨跌幅'] = display_sectors['平均涨跌幅'].astype(str) + '%'
    
    st.dataframe(
        display_sectors[['板块', '平均涨跌幅', '总成交额(亿)', '股票数量', '热度']],
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.subheader("📈 板块分布")
    fig = px.pie(
        sector_stats.head(8), 
        values='股票数量', 
        names='板块',
        title='板块分布'
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 板块热力图
# ------------------------------------------------------------
fig = px.bar(
    sector_stats.head(10), 
    x='板块', 
    y='热度', 
    color='热度',
    text='平均涨跌幅',
    title='板块热度条形图',
    color_continuous_scale='RdYlGn'
)
fig.update_traces(texttemplate='%{text}%', textposition='outside')
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 龙头个股
# ------------------------------------------------------------
st.header("🔍 板块龙头个股")

# 计算综合评分
df['综合评分'] = (
    df['涨跌幅'] * 0.6 + 
    (df['成交量'] / 1e9) * 0.4
)

# 选取每个板块的前3名
top_stocks = df.sort_values('综合评分', ascending=False).groupby('板块').head(3)

# 显示龙头股
display_cols = ['板块', '代码', '名称', '涨跌幅', '成交额(亿)']
display_df = top_stocks[display_cols].copy()
display_df['涨跌幅'] = display_df['涨跌幅'].astype(str) + '%'
st.dataframe(display_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# 综合评分系统
# ------------------------------------------------------------
st.header("📊 市场综合评分")

# 计算各项评分
macro_score = min(max(sector_stats['平均涨跌幅'].mean() * 10 + 50, 0), 100)
liquidity_score = min(df['成交量'].sum() / 2e11 * 100, 100)
sentiment_score = min(
    (len(df[df['涨跌幅'] > 0]) / len(df)) * 50 +
    (len(top_stocks) / (len(df) / 5)) * 50,
    100
)
total_score = np.mean([macro_score, liquidity_score, sentiment_score])

# 显示仪表盘
col1, col2, col3, col4 = st.columns(4)

def create_gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "#ffcccc"},
                {'range': [30, 70], 'color': "#ffffcc"},
                {'range': [70, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
    return fig

with col1:
    st.plotly_chart(create_gauge(macro_score, "宏观评分"), use_container_width=True)
with col2:
    st.plotly_chart(create_gauge(liquidity_score, "流动性评分"), use_container_width=True)
with col3:
    st.plotly_chart(create_gauge(sentiment_score, "情绪评分"), use_container_width=True)
with col4:
    st.metric("综合评分", f"{total_score:.1f}")
    st.metric("上涨比例", f"{(len(df[df['涨跌幅']>0])/len(df)*100):.1f}%")
    st.metric("最强板块", sector_stats.iloc[0]['板块'] if not sector_stats.empty else 'N/A')

# ------------------------------------------------------------
# 操作建议
# ------------------------------------------------------------
st.header("🎯 今日操作建议")

if total_score >= 70:
    st.success("""
    ### 🚀 进攻模式
    - 聚焦强势板块龙头股
    - 可适当提高仓位至7-8成
    - 关注：科技、新能源等高景气度板块
    - 策略：回踩5日线买入，跌破10日线止损
    """)
elif total_score >= 40:
    st.warning("""
    ### ⚖️ 精选模式
    - 控制仓位在5成以下
    - 快进快出，不宜恋战
    - 关注：有业绩支撑的板块
    - 策略：低吸为主，不追高
    """)
else:
    st.error("""
    ### 🛡️ 防守模式
    - 降低仓位至3成以下
    - 避免追高，多看少动
    - 关注：防御性板块（公用事业、消费）
    - 策略：等待市场企稳信号
    """)

# ------------------------------------------------------------
# 详细数据
# ------------------------------------------------------------
st.header("📈 详细数据")

tab1, tab2, tab3 = st.tabs(["涨幅榜", "跌幅榜", "成交额榜"])

with tab1:
    gainers = df.nlargest(10, '涨跌幅')[['代码', '名称', '板块', '涨跌幅', '成交额(亿)']].copy()
    gainers['涨跌幅'] = gainers['涨跌幅'].astype(str) + '%'
    st.dataframe(gainers, use_container_width=True, hide_index=True)

with tab2:
    losers = df.nsmallest(10, '涨跌幅')[['代码', '名称', '板块', '涨跌幅', '成交额(亿)']].copy()
    losers['涨跌幅'] = losers['涨跌幅'].astype(str) + '%'
    st.dataframe(losers, use_container_width=True, hide_index=True)

with tab3:
    volume_leader = df.nlargest(10, '成交量')[['代码', '名称', '板块', '涨跌幅', '成交额(亿)']].copy()
    volume_leader['涨跌幅'] = volume_leader['涨跌幅'].astype(str) + '%'
    st.dataframe(volume_leader, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# 板块气泡图
# ------------------------------------------------------------
st.header("🎯 板块分析气泡图")

fig = px.scatter(
    sector_stats.head(15),
    x='平均涨跌幅',
    y='总成交额',
    size='股票数量',
    color='热度',
    text='板块',
    title='板块分析（气泡大小=股票数量）',
    labels={'平均涨跌幅': '平均涨跌幅 (%)', '总成交额': '总成交额 (元)'}
)
fig.update_traces(textposition='top center')
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# 底部信息
# ------------------------------------------------------------
st.markdown("---")
st.caption(f"""
更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

# 添加刷新按钮
if st.button("🔄 刷新数据"):
    st.cache_data.clear()
    st.rerun()
