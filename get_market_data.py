import time
from datetime import date, timedelta

import yfinance as yf

import pandas as pd

def get_trading_days_before(end_date, trading_days):
    """Calculates the date that is a specified number of trading days before a given date."""
    current_date = end_date
    days_counted = 0
    while days_counted < trading_days:
        current_date -= timedelta(days=1)
        if current_date.weekday() < 5:
            days_counted += 1
    return current_date

    
stocks = ["MSFT", "GOOGL", "AMZN", "META", "BIDU", "NVDA", "AMD", "TSM", "CRM", "ADBE"]
market_index = "^GSPC"

event_date = date(2025, 1, 20)
start_date = get_trading_days_before(event_date, 151)
end_date = get_trading_days_before(event_date, 30)

stock_data = pd.DataFrame()

for stock in stocks:
    df = yf.download(stock, start=start_date, end=end_date)

    if "Adj Close" in df.columns:
        close_col = "Adj Close"
    elif "Close" in df.columns:
        close_col = "Close"
    else:
        print(f"{stock} has no 'Close' or 'Adj Close', skipped")
        continue

    df = df[[close_col, "Volume"]].copy()
    df.rename(columns={close_col: f"{stock}_Close", "Volume": f"{stock}_Volume"}, inplace=True)
    df[f"{stock}_Return"] = df[f"{stock}_Close"].pct_change()

    if stock_data.empty:
        stock_data = df
    else:
        stock_data = stock_data.join(df, how="outer")

# S&P 500
market_df = yf.download(market_index, start=start_date, end=end_date)
if "Adj Close" in market_df.columns:
    market_df.rename(columns={"Adj Close": "SP500_Close"}, inplace=True)
elif "Close" in market_df.columns:
    market_df.rename(columns={"Close": "SP500_Close"}, inplace=True)
else:
    raise ValueError("missing market index data")

market_df["SP500_Return"] = market_df["SP500_Close"].pct_change()

# merge data
stock_data = stock_data.join(market_df, how="outer")

# save as csv
stock_data.to_csv("market_info.csv")

print("data downloaded, saved as 'market_info.csv'")