import yfinance as yf
import pandas as pd

# 定义SP500指数代码和时间范围
sp500_ticker = "^GSPC"  # SP500指数的股票代码
start_date = "2024-08-09"
end_date = "2025-02-08"

# 下载SP500指数的历史数据
print(f"正在下载 {sp500_ticker} 数据...")
sp500_data = yf.download(sp500_ticker, start=start_date, end=end_date)

# 检查是否成功下载数据
if not sp500_data.empty:
    print(f"成功下载 {sp500_ticker} 的数据.")

    # 计算每日回报率 (百分比变化)
    sp500_data['MarketReturn'] = sp500_data['Close'].pct_change() * 100  # 计算收盘价的百分比变化

    # 创建一个包含Date和MarketReturn的DataFrame
    market_return_df = pd.DataFrame(sp500_data['MarketReturn']).dropna()  #  dropna() 移除第一天由于pct_change产生的NaN值
    market_return_df.index.name = 'Date'  # 将索引列命名为 'Date'

    # 打印 DataFrame 的 head
    print("\nSP500指数回报率 DataFrame 的 head:")
    print(market_return_df.head())

    # 将 DataFrame 保存到 CSV 文件
    csv_filename = "../data/SP500_MarketReturn.csv"
    market_return_df.to_csv(csv_filename)
    print(f"\nSP500指数回报率数据已保存到文件: {csv_filename}")

else:
    print(f"无法下载 {sp500_ticker} (SP500指数) 的数据.")