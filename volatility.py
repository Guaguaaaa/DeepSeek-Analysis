import yfinance as yf
import pandas as pd
import numpy as np

# 定义股票列表
stocks = ["MSFT", "GOOGL", "AMZN", "META", "BIDU", "NVDA", "AMD", "TSM", "CRM", "ADBE"]

# 设置时间范围
start_date = "2024-07-29"
end_date = "2025-02-08"

# 下载股票数据
data = yf.download(stocks, start=start_date, end=end_date)["Close"]

# 计算每日收益率
returns = data.pct_change()

# 计算滚动窗口的历史波动率（每日计算，窗口设为5天）
rolling_volatility = returns.rolling(window=10).std() * np.sqrt(252)

# 打印结果
print(rolling_volatility)

rolling_volatility.to_csv("rolling_volatility.csv")