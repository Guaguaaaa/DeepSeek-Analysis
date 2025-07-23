import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # 指定后端为 TkAgg
import matplotlib.pyplot as plt

# 读取CSV数据
stock_data = pd.read_csv("../data/Stock_Herding_Deepseek.csv")

# 确保'Date'列是datetime格式
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# 筛选1月1日至2月28日的数据，并创建副本避免警告
filtered_data = stock_data[(stock_data['Date'] >= '2024-08-12') & (stock_data['Date'] <= '2025-02-08')].copy()

# 获取所有股票的收盘价列
stocks = [col for col in filtered_data.columns if '_Close' in col]

# 计算所有股票的日收益率
for stock in stocks:
    filtered_data[stock + '_Return'] = filtered_data[stock].pct_change()

# 计算每日CSSD（横截面标准差）
filtered_data['CSSD'] = filtered_data[[stock + '_Return' for stock in stocks]].std(axis=1)

# 查看计算结果
print(filtered_data[['Date', 'CSSD']].head())

# 保存计算结果
filtered_data.to_csv('Filtered_Stock_Data_with_CSSD.csv', index=False)

plt.hist(filtered_data['CSSD'].dropna(), bins=30, edgecolor='black')
plt.title('Distribution of CSSD')
plt.xlabel('CSSD')
plt.ylabel('Frequency')
plt.show()