import yfinance as yf
import pandas as pd
import time

# 股票列表
stocks = ["MSFT", "GOOGL", "AMZN", "META", "BIDU", "NVDA", "AMD", "TSM", "CRM", "ADBE"]

# 获取数据的时间范围
start_date = "2024-08-12"
end_date = "2025-02-08"

# 创建一个空的数据框来保存所有股票的换手率数据
turnover_data = pd.DataFrame()

# 计算换手率
for stock in stocks:
    try:
        # 获取股票数据
        ticker = yf.Ticker(stock)

        # 获取股票的历史交易数据（包括成交量）
        data = ticker.history(start=start_date, end=end_date)

        # 获取流通股数（此数据通常存储在info字段中）
        stock_info = ticker.info
        shares_outstanding = stock_info.get('sharesOutstanding', None)

        if shares_outstanding is not None:
            # 计算换手率：成交量 / 流通股数
            data['Turnover_Ratio'] = data['Volume'] / shares_outstanding

            # 保留股票名称和换手率数据
            turnover_data[stock] = data['Turnover_Ratio']
        else:
            print(f"无法获取{stock}的流通股数数据。")

        # 每次请求完一只股票后，等待一段时间（防止触发速率限制）
        time.sleep(5)  # 每次请求后延迟5秒

    except yf.exceptions.YFRateLimitError:
        print("Rate limit exceeded. Retrying after 60 seconds.")
        time.sleep(60)  # 如果触发速率限制，等待60秒再重试

# 显示换手率数据
print(turnover_data.head())

# 如果需要保存数据为CSV文件
turnover_data.to_csv('turnover_ratios.csv')
