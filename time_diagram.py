import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def mean_diagram(df):
    # 确保Date列为datetime格式
    df["Date"] = pd.to_datetime(df["Date"])

    # 计算每个日期的Price平均值
    avg_price_per_day = df.groupby("Date")["Price"].mean()

    # 创建图形
    plt.figure(figsize=(12, 6))
    plt.plot(avg_price_per_day.index, avg_price_per_day.values, label="Price")

    # 添加2025-01-20的垂直红色虚线
    highlight_date = pd.to_datetime("2025-01-20")
    plt.axvline(x=highlight_date, color="red", linestyle="--", label="20/01/2025")

    # 设置x轴日期格式为DD/MM/YYYY
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gcf().autofmt_xdate()

    # 图形美化
    plt.xlabel("Date")
    plt.ylabel("Average Price")
    plt.title("Average Stock Price Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 显示图形
    plt.savefig('..\\Diagram\\Price.png', bbox_inches='tight')
    plt.show()


def single_diagram(df):
    # 确保Date列为datetime格式
    df['Date'] = pd.to_datetime(df['Date'])

    # 仅选择NVDA的股票数据
    nvda_data = df[df['Stocks'].str.strip().str.lower() == 'nvda']

    # 创建图表和子图
    fig, ax = plt.subplots(figsize=(12, 6))

    # 画出NVDA股票的线型图（不加label以免图例出现“NVDA”）
    ax.plot(nvda_data['Date'], nvda_data['Price'])

    # 高亮三个关键日期
    highlight_dates = [
        (datetime(2025, 1, 20), '-- 20/01/2025', 'red', 'vline'),  # 虚线
        (datetime(2025, 1, 23), '● 23/01/2025', 'red', 'dot'),  # 红点
        (datetime(2025, 1, 27), '● 27/01/2025', 'red', 'dot')  # 红点
    ]

    for date, label, color, kind in highlight_dates:
        if kind == 'vline':
            ax.axvline(date, color=color, linestyle='--', linewidth=1.5, label=label)
        elif kind == 'dot':
            # 找到当天的价格
            price_row = nvda_data[nvda_data['Date'] == date]
            if not price_row.empty:
                price = price_row['Price'].values[0]
                ax.plot(date, price, marker='o', color=color, label=label)

    # 格式化x轴为DD/MM/YYYY格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    fig.autofmt_xdate()

    # 添加图例和标签
    ax.set_title('Daily Stock Price Fluctuation')
    ax.set_xlabel('Date')
    ax.set_ylabel('NVDA Close Price')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


# 读取CSV文件
df = pd.read_csv("../data/final_data/new_30_data.csv")

# single_diagram(df)

mean_diagram(df)