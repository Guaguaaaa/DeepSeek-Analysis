import yfinance as yf
import pandas as pd
import numpy as np

# 定义股票代码列表和时间周期
tickers = ["MSFT", "GOOGL", "AMZN", "META", "BIDU", "NVDA", "AMD", "TSM", "CRM", "ADBE"]
start_date = "2024-07-30"
end_date = "2025-02-08"
wr_period = 10  # WR计算的时间周期为过去10天

# 定义计算威廉姆斯百分比范围(WR)的函数
def calculate_wr(data, period=10):
    """
    计算WR

    参数:
    data (pd.DataFrame): 包含 'High', 'Low', 'Close' 列的股票数据DataFrame
    period (int): WR计算的时间周期，默认值为10

    返回:
    pd.Series: WR 值序列
    """
    highest_high = data['High'].rolling(window=period).max() # 计算过去period日最高价的最大值
    lowest_low = data['Low'].rolling(window=period).min()   # 计算过去period日最低价的最小值
    wr = (highest_high - data['Close']) / (highest_high - lowest_low) * -100 # WR公式
    return wr

# 创建一个空的DataFrame来存储所有股票的数据
df = pd.DataFrame()
date_range = pd.to_datetime(pd.date_range(start=start_date, end=end_date)) # 生成日期范围用于索引
df['Date'] = date_range
df = df.set_index('Date') # 将Date设置为索引

# 循环遍历每支股票，下载数据并计算WR
for ticker in tickers:
    print(f"正在下载 {ticker} 的数据...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    if not stock_data.empty: # 检查是否成功下载数据
        print(f"正在计算 {ticker} 的WR...")
        wr_values = calculate_wr(stock_data, period=wr_period) # 计算WR值
        wr_values.name = 'WR' # 直接设置Series的name属性为 'WR'


        # 将 High, Low, Close 和 WR 值添加到主 DataFrame 中
        df[f'{ticker}_high'] = stock_data['High']
        df[f'{ticker}_low'] = stock_data['Low']
        df[f'{ticker}_close'] = stock_data['Close']
        df[f'{ticker}_WR'] = wr_values

        # 计算 WR_Index，根据WR值判断是否超出阈值
        wr_index = np.where((df[f'{ticker}_WR'] > -20) | (df[f'{ticker}_WR'] < -80), 1, 0) # 使用 numpy.where 实现条件判断，符合条件为1，否则为0
        df[f'{ticker}_WR_Index'] = wr_index

    else:
        print(f"无法下载 {ticker} 的数据，跳过.")

# 移除在计算WR时由于rolling period产生的NaN值
df = df.dropna() # 移除包含NaN的行，主要是因为rolling计算初期数据不足period天会产生NaN

# 打印 DataFrame 的 head
print("\nDataFrame 的 head:")
print(df.head())

# 将 DataFrame 保存到 CSV 文件
csv_filename = "../data/WR_data.csv"
df.to_csv(csv_filename)
print(f"\n数据已保存到文件: {csv_filename}")