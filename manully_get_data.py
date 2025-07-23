import pandas as pd
import os

tickers = ["msft", "googl", "amzn", "meta", "bidu", "nvda", "amd", "tsm", "crm", "adbe"]
dataframes = {}

def clean_price(value):
    """Removes '$' and converts to float."""
    if isinstance(value, str) and '$' in value:
        return float(value.replace('$', ''))
    return value

for ticker in tickers:
    filename = f"../seperate_data\\market_data\\{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df.set_index('Date', inplace=True)

        rename_dict = {
            "Close/Last": f"{ticker}_Close",
            "Volume": f"{ticker}_Volume",
            "Open": f"{ticker}_Open",
            "High": f"{ticker}_High",
            "Low": f"{ticker}_Low",
        }
        df.rename(columns=rename_dict, inplace=True)

        # Apply cleaning to price columns
        for col in [f"{ticker}_Close", f"{ticker}_Open", f"{ticker}_High", f"{ticker}_Low"]:
            df[col] = df[col].apply(clean_price)

        dataframes[ticker] = df

    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Handle the sp500.csv file
sp500_filename = "../seperate_data\\market_data\\sp500.csv"
try:
    sp500_df = pd.read_csv(sp500_filename)
    sp500_df['Date'] = pd.to_datetime(sp500_df['Date'], format='%m/%d/%Y')
    sp500_df.set_index('Date', inplace=True)
    sp500_df.rename(columns={"Close/Last": "sp500_Close", "Open": "sp500_Open", "High": "sp500_High", "Low":"sp500_Low"}, inplace=True)

    for col in ["sp500_Close", "sp500_Open", "sp500_High", "sp500_Low"]:
        sp500_df[col] = sp500_df[col].apply(clean_price)

    dataframes['sp500'] = sp500_df
except FileNotFoundError:
    print(f"File not found: {sp500_filename}")
except Exception as e:
    print(f"Error processing {sp500_filename}: {e}")

merged_df = pd.concat(dataframes.values(), axis=1)

merged_df.sort_index(inplace=True)
merged_df.index = merged_df.index.strftime('%Y-%m-%d')

merged_df.to_csv("full_data.csv")

print("Merged data saved to full_data.csv")