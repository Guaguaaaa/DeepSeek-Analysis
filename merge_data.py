import pandas as pd
import os

tickers = ["msft", "googl", "amzn", "meta", "bidu", "nvda", "amd", "tsm", "crm", "adbe"]
dataframes = {}

def parse_date(date_str):
    """Parses date strings with different formats."""
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    except ValueError:
        return pd.to_datetime(date_str, format='%d/%m/%Y')

for ticker in tickers:
    filename = f"temp_data\\reddit_mention\\{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df['Date'] = df['Date'].apply(parse_date)  # Parse dates using custom function
        df.set_index('Date', inplace=True)
        df.rename(columns={"TSM": ticker}, inplace=True)
        dataframes[ticker] = df[ticker]
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

if dataframes:
    merged_df = pd.concat(dataframes.values(), axis=1, join='outer')
    merged_df.sort_index(inplace=True)
    merged_df.index = merged_df.index.strftime('%Y-%m-%d')
    merged_df.to_csv("full_reddit_mention.csv")
    print("Merged twitter mention data saved to full_reddit_mention.csv")
else:
    print("No twitter mention files were found, so no file was created.")