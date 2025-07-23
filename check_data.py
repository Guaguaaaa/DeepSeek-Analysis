import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('TkAgg')  # 指定后端为 TkAgg
import matplotlib.pyplot as plt

# The following line loads the confidential dataset, which is not included in this repository.
# Therefore, this line has been commented out.
# stock_data = pd.read_csv("data/Stock_Herding_Deepseek.csv", index_col="Date", parse_dates=True)

'''
# 读取 CSV 文件，先不指定 index_col
df = pd.read_csv("Stock_Herding_Data.csv")

# 输出前 5 行数据
print(df.head())

# 输出列名
print("\n列名：", df.columns)
'''
# 股票列表
stocks = ["MSFT", "GOOGL", "AMZN", "META", "BIDU", "NVDA", "AMD", "TSM", "CRM", "ADBE"]

# 查看极值
print("🔎 查看每只股票的最大值和最小值：\n")

for stock in stocks:
    print(f"\n{stock} 的极值：")
    print(f"最大收盘价：{stock_data[f'{stock}_Close'].max():.2f} 日期: {stock_data[f'{stock}_Close'].idxmax()}")
    print(f"最小收盘价：{stock_data[f'{stock}_Close'].min():.2f} 日期: {stock_data[f'{stock}_Close'].idxmin()}")
    print(f"最大交易量：{stock_data[f'{stock}_Volume'].max()} 日期: {stock_data[f'{stock}_Volume'].idxmax()}")
    print(f"最小交易量：{stock_data[f'{stock}_Volume'].min()} 日期: {stock_data[f'{stock}_Volume'].idxmin()}")
    print(f"最大收益率：{stock_data[f'{stock}_Return'].max():.4f} 日期: {stock_data[f'{stock}_Return'].idxmax()}")
    print(f"最小收益率：{stock_data[f'{stock}_Return'].min():.4f} 日期: {stock_data[f'{stock}_Return'].idxmin()}")
    print("-" * 60)

# 绘制图表
plt.figure(figsize=(12, 10))

for i, stock in enumerate(stocks):
    plt.subplot(5, 2, i + 1)
    plt.plot(stock_data.index, stock_data[f'{stock}_Close'], label='Close', color='blue', alpha=0.7)
    plt.title(f"{stock} - Close Price")
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 绘制交易量和收益率的图表
plt.figure(figsize=(12, 10))

for i, stock in enumerate(stocks):
    plt.subplot(5, 2, i + 1)
    plt.plot(stock_data.index, stock_data[f'{stock}_Volume'], label='Volume', color='green', alpha=0.7)
    plt.title(f"{stock} - Trading Volume")
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))

for i, stock in enumerate(stocks):
    plt.subplot(5, 2, i + 1)
    plt.plot(stock_data.index, stock_data[f'{stock}_Return'], label='Return', color='red', alpha=0.7)
    plt.title(f"{stock} - Daily Return")
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
