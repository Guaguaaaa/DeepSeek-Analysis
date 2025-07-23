import yfinance as yf
import pandas as pd
import time

# 定义研究股票列表
stocks = ["MSFT", "GOOGL", "AMZN", "META", "BIDU", "NVDA", "AMD", "TSM", "CRM", "ADBE"]
market_index = "^GSPC"

start_date = "2024-07-15"
end_date = "2025-03-03"

stock_data = pd.DataFrame()

for stock in stocks:
    while True:  # 循环直到成功下载
        try:
            df = yf.download(stock, start=start_date, end=end_date)

            # 确保列名正确
            if "Adj Close" in df.columns:
                close_col = "Adj Close"
            elif "Close" in df.columns:
                close_col = "Close"
            else:
                print(f"⚠️ {stock} 没有 'Close' 或 'Adj Close'，跳过")
                break  # 跳出内层while循环，处理下一只股票

            df = df[[close_col, "Volume"]].copy()
            df.rename(columns={close_col: f"{stock}_Close", "Volume": f"{stock}_Volume"}, inplace=True)
            df[f"{stock}_Return"] = df[f"{stock}_Close"].pct_change()

            if stock_data.empty:
                stock_data = df
            else:
                stock_data = stock_data.join(df, how="outer")
            break  # 下载成功，跳出内层while循环
        except Exception as e:
            print(f"⚠️ 下载 {stock} 时出错: {e}. 60秒后重试...")
            time.sleep(60)

# 下载市场指数数据（S&P 500）
while True:
    try:
        market_df = yf.download(market_index, start=start_date, end=end_date)
        if "Adj Close" in market_df.columns:
            market_df.rename(columns={"Adj Close": "SP500_Close"}, inplace=True)
        elif "Close" in market_df.columns:
            market_df.rename(columns={"Close": "SP500_Close"}, inplace=True)
        else:
            raise ValueError("⚠️ 市场指数数据缺失")

        market_df["SP500_Return"] = market_df["SP500_Close"].pct_change()

        # 合并市场数据
        stock_data = stock_data.join(market_df, how="outer")

        break #Market Data download success.
    except Exception as e:
        print(f"⚠️ 下载市场指数时出错: {e}. 60秒后重试...")
        time.sleep(60)

# 保存为 CSV 文件
stock_data.to_csv("Stock_Herding_Deepseek.csv")

print("✅ 数据下载完成，已保存为 'Stock_Herding_Deepseek.csv'")