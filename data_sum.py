import pandas as pd
import numpy as np

# 定义股票代码列表和日期范围 (确保与你的文件中的列名一致)
tickers = ["MSFT", "GOOGL", "AMZN", "META", "BIDU", "NVDA", "AMD", "TSM", "CRM", "ADBE"]
start_date = "2024-08-12"
end_date = "2025-02-07"

# 加载 CSV 文件 (假设文件都保存在当前工作目录下，如果不是，请提供完整的文件路径)
try:
    wr_data = pd.read_csv("../data/WR_data.csv", index_col="Date", parse_dates=True)

    turnover_ratios_data = pd.read_csv("../data/turnover_ratios.csv", parse_dates=['Date'])  # 先不设置 Date 为 index
    turnover_ratios_data['Date'] = pd.to_datetime(turnover_ratios_data['Date'], utc=True).dt.tz_localize(
        None).dt.date  # 转换 Date 列为只有日期的格式, 并去除时区信息
    turnover_ratios_data = turnover_ratios_data.set_index('Date')  # 重新将 Date 列设置为 index

    rolling_volatility_data = pd.read_csv("../data/rolling_volatility.csv", index_col="Date", parse_dates=True)
    sp500_market_return_data = pd.read_csv("../data/SP500_MarketReturn.csv", index_col="Date", parse_dates=True)
    fomo_sentiment_data = pd.read_csv("../data/FOMO sentiment.csv", index_col="Date", parse_dates=True)
    print("CSV 文件加载成功.")
except FileNotFoundError as e:
    print(f"文件加载失败，请检查文件路径是否正确: {e}")
    exit()


# 初始化新的 DataFrame 用于存储整合后的数据
target_date_index = sp500_market_return_data.index
wr_data = wr_data.reindex(target_date_index)
turnover_ratios_data = turnover_ratios_data.reindex(target_date_index)
rolling_volatility_data = rolling_volatility_data.reindex(target_date_index)
fomo_sentiment_data = fomo_sentiment_data.reindex(target_date_index)

df_sum = pd.DataFrame(index=sp500_market_return_data.index) # 使用原始 DataFrame 的日期索引
df_sum.index.name = 'Date'

# 计算 HerdingIndicator_t 的每日平均值
herding_indicator_cols = [f'{ticker}_WR' for ticker in tickers]
df_sum['HerdingIndicator_t'] = wr_data[herding_indicator_cols].mean(axis=1)
print("HerdingIndicator_t 平均值计算完成.")
df_sum['HerdingIndicator_t'] = np.where(
    (df_sum['HerdingIndicator_t'] > -20) | (df_sum['HerdingIndicator_t'] < -80),
    1,  # 条件为真时，赋值为 1
    0   # 条件为假时，赋值为 0
)
print("HerdingIndicator_t 列已转换为二元指标 (0或1).")

# 计算 OverConfidencet (Turnover Ratio) 的每日平均值
overconfidence_cols = tickers # turnover_ratios.csv 的列名直接是股票代码
df_sum['OverConfidencet'] = turnover_ratios_data[overconfidence_cols].mean(axis=1)
print("OverConfidencet 平均值计算完成.")

# 计算 Volatility_t 的每日平均值
volatility_cols = tickers # rolling_volatility.csv.csv 的列名直接是股票代码
df_sum['Volatility_t'] = rolling_volatility_data[volatility_cols].mean(axis=1)
print("Volatility_t 平均值计算完成.")

# 获取 FOMO_t (Sentiment) 的每日平均值
fomo_cols = tickers # FOMO sentiment.csv 的列名直接是股票代码
df_sum['FOMO_t'] = fomo_sentiment_data[fomo_cols].mean(axis=1) / 100
print("FOMO_t 平均值计算完成.")

# 直接复制 MarketReturn_t (S&P500 Market Return)
df_sum['MarketReturn_t'] = sp500_market_return_data['MarketReturn']
print("MarketReturn_t 数据复制完成.")

# 移除所有包含 NaN 的行
#df_sum = df_sum.dropna()
#print("移除 NaN 值完成.")

# 打印 DataFrame 的 head
print("\n整合后的 DataFrame 的 head:")
print(df_sum.head())

# 将 DataFrame 保存到 CSV 文件
csv_filename = "../data/data_sum.csv"
df_sum.to_csv(csv_filename)
print(f"\n整合后的数据已保存到文件: {csv_filename}")