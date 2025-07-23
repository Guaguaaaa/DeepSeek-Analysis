import pandas as pd
import numpy as np


def filter_date(df):
    # Convert 'Date' to datetime objects for filtering
    df['Date'] = pd.to_datetime(df['Date'])

    # Define the date range
    start_date = pd.to_datetime("2024-12-04")
    end_date = pd.to_datetime("2025-03-04")

    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    return filtered_df


def cssd(tickers):
    df = pd.read_csv('full_stock_data.csv')

    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    except ValueError:
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    # calculate return
    for ticker in tickers:
        df[f'{ticker}_Return'] = df[f'{ticker}_Close'].pct_change()
        df['sp500_Return'] = df['sp500_Close'].pct_change()

    df = filter_date(df)

    # calculate cssd
    rm_column = 'sp500_Return'
    unique_dates = df['Date'].unique()

    cssd_data = []
    for date in unique_dates:
        for ticker in tickers:
            row = df[df['Date'] == date].iloc[0]
            rj_values = row[[f'{t}_Return' for t in tickers if t != ticker]].values
            rm_value = row[rm_column]

            squared_diff_sum = np.sum((rj_values - rm_value)**2)
            cssd_value = np.sqrt(squared_diff_sum / (len(tickers) - 2))

            cssd_data.append({'Date': date, 'Stocks': ticker, 'CSSD': cssd_value})
    cssd_df = pd.DataFrame(cssd_data)

    cssd_df.to_csv("new_30_data.csv", index=False)
    print("CSSD saved to new_30_data.csv")


def calculate_wr(data, period):
    highest_high = data['High'].rolling(window=period).max()
    lowest_low = data['Low'].rolling(window=period).min()
    wr = (highest_high - data['Close']) / (highest_high - lowest_low) * -100
    return wr


def wr(tickers):
    df = pd.read_csv('full_stock_data.csv')
    new_var_data_df = pd.read_csv("new_30_data.csv")

    wr_period = 10

    for ticker in tickers:
        data = df[['Date', f'{ticker}_Close', f'{ticker}_High', f'{ticker}_Low']].copy()
        data.rename(columns={f'{ticker}_Close' : 'Close', f'{ticker}_High' : 'High', f'{ticker}_Low' : 'Low'}, inplace=True)
        wr_values = calculate_wr(data, period=wr_period)
        wr_values.name = 'WR'

        df[f'{ticker}_WR'] = wr_values

        wr_index = np.where((df[f'{ticker}_WR'] > -20) | (df[f'{ticker}_WR'] < -80), 1, 0)
        df[f'{ticker}_WR_Index'] = wr_index

    # print("successfully calculated WR")

    df = df.dropna()

    wr_column_mapping = {ticker: f'{ticker}_WR_Index' for ticker in tickers}

    new_var_data_df['WR'] = np.nan

    for index, row in new_var_data_df.iterrows():
        date = row['Date']
        stock = row['Stocks']

        wr_row = df[df['Date'] == date]
        if not wr_row.empty:
            wr_value = wr_row[wr_column_mapping[stock]].iloc[0]
            new_var_data_df.loc[index, 'WR'] = wr_value

    new_var_data_df.to_csv("new_30_data.csv", index=False)
    print("WR saved to new_30_data.csv")


def trading_volume(tickers):
    df = pd.read_csv("full_stock_data.csv")
    new_var_df = pd.read_csv("new_30_data.csv")
    volume_col_mapping = {ticker: f'{ticker}_Volume' for ticker in tickers}

    new_var_df['TradingVolume'] = np.nan

    for index, row in new_var_df.iterrows():
        date = row['Date']
        stock = row['Stocks']

        volume_row = df[df['Date'] == date]
        if not volume_row.empty:
            volume_value = volume_row[volume_col_mapping[stock]].iloc[0]
            new_var_df.loc[index, 'TradingVolume'] = volume_value

    new_var_df.to_csv("new_30_data.csv", index=False)
    print("Trading Volume saved to new_30_data.csv")


def turnover_ratio(tickers):
    share_outstanding = {
        'nvda' : 24400000000,
        'msft' : 7430000000,    # 7433982235
        'amd' : 1620000000,    #1620477962
        'bidu' : 360700000,
        'tsm' : 5190000000,
        'crm' : 961000000,
        'amzn' : 10600000000,
        'meta' : 2190000000,
        'adbe' : 434900000,
        'googl' : 5830000000
    }
    df = pd.read_csv("full_stock_data.csv")
    data = df[['Date']].copy()

    for ticker in tickers:
        data[ticker] = df[f'{ticker}_Volume'] / share_outstanding[ticker]

    # data = data.set_index('Date')

    new_var_df = pd.read_csv("new_30_data.csv")
    turnover_col_mapping = {ticker: ticker for ticker in tickers}
    new_var_df['TurnoverRatio'] = np.nan

    for index, row in new_var_df.iterrows():
        date = row['Date']
        stock = row['Stocks']

        turnover_row = data[data['Date'].str.startswith(date)]
        if not turnover_row.empty:
            turnover_value = turnover_row[turnover_col_mapping[stock]].iloc[0]
            new_var_df.loc[index, 'TurnoverRatio'] = turnover_value

    new_var_df.to_csv("new_30_data.csv", index=False)
    print("turnover ratio saved to new_30_data.csv")


def rolling_volatility(tickers):
    df = pd.read_csv("full_stock_data.csv")
    rv_df = df[['Date']].copy()
    for ticker in tickers:
        returns = df[f'{ticker}_Close'].pct_change()
        rolling_volatility = returns.rolling(window=10).std() * np.sqrt(252)

        rv_df[f'{ticker}'] = rolling_volatility
    rv_df = rv_df.dropna()

    rv_df = rv_df.set_index('Date')


    new_var_df = pd.read_csv("new_30_data.csv")

    rv_long_df = rv_df.stack().reset_index()
    rv_long_df.columns = ['Date', 'Stocks', 'RollingVolatility']
    rv_long_df.to_csv("temp.csv")

    new_var_df = pd.merge(new_var_df, rv_long_df, on=['Date', 'Stocks'], how='left')

    new_var_df.to_csv("new_30_data.csv", index=False)
    print("Rolling Volume saved to new_30_data.csv")


def sentiment(tickers):
    df_twitter = pd.read_csv("data/full_twitter_mention.csv")
    df_reddit = pd.read_csv("data/full_reddit_mention.csv")

    df_twitter = filter_date(df_twitter)
    df_reddit = filter_date(df_reddit)

    df_twitter = df_twitter.set_index('Date')
    df_reddit = df_reddit.set_index('Date')

    df_twitter.index = pd.to_datetime(df_twitter.index, format='%Y-%m-%d')
    df_reddit.index = pd.to_datetime(df_reddit.index, format='%Y-%m-%d')

    common_dates = df_twitter.index.intersection(df_reddit.index)
    df_twitter = df_twitter.loc[common_dates]
    df_reddit = df_reddit.loc[common_dates]

    for col in tickers:
        df_twitter[col] = df_twitter[col].interpolate(method='linear')
        df_reddit[col] = df_reddit[col].interpolate(method='linear')

    missing_dates = df_twitter[df_twitter.isnull().any(axis=1)].index

    min_val_twi = df_twitter[tickers].min().min()
    max_val_twi = df_twitter[tickers].max().max()
    min_val_reddit = df_reddit[tickers].min().min()
    max_val_reddit = df_reddit[tickers].max().max()

    for col in tickers:
        df_twitter[col] = ((df_twitter[col] - min_val_twi) / (max_val_twi - min_val_twi)) * 100
        df_reddit[col] = ((df_reddit[col] - min_val_reddit) / (max_val_reddit - min_val_reddit)) * 100

    df_sen_adj = pd.DataFrame(index=common_dates)
    for col in tickers:
        df_sen_adj[col] = (df_twitter[col] + df_reddit[col]) / 2

    df_sen_adj_long = df_sen_adj.stack().reset_index()
    df_sen_adj_long.columns = ['Date', 'Stocks', 'Sentiment']

    df_sen_adj_long['Date'] = pd.to_datetime(df_sen_adj_long['Date'])
    df_sen_adj_long['Date'] = df_sen_adj_long['Date'].dt.strftime('%Y-%m-%d')

    new_var_df = pd.read_csv("new_30_data.csv")
    new_var_df = pd.merge(new_var_df, df_sen_adj_long, on=['Date', 'Stocks'], how='left')

    new_var_df.to_csv("new_30_data.csv", index=False)
    print("Sentiment (Adjust) saved to new_30_data.csv")


def market_return():
    df = pd.read_csv("full_stock_data.csv", usecols=['Date', 'sp500_Close'])
    df['sp500_Return'] = df['sp500_Close'].pct_change()
    df = df.set_index('Date')
    df = df.drop('sp500_Close', axis=1)

    df.rename(columns={'sp500_Return': 'MarketReturn'}, inplace=True)

    new_var_df = pd.read_csv("new_30_data.csv")
    new_var_df = pd.merge(new_var_df, df, on='Date', how='left')

    new_var_df.to_csv("new_30_data.csv", index=False)
    print("Market Return saved to new_30_data.csv")


def vix():
    df = pd.read_csv("data/VIXCLS.csv", usecols=['Date', 'VIXCLS'])
    df = df.set_index('Date')
    df.rename(columns={'VIXCLS': 'VIX'}, inplace=True)

    new_var_df = pd.read_csv("new_30_data.csv")
    new_var_df = pd.merge(new_var_df, df, on='Date', how='left')

    new_var_df.to_csv("new_30_data.csv", index=False)
    print("VIXCLS saved to new_30_data.csv")


def bwbp(tickers):
    """
    计算给定股票列表中所有股票的布林带宽度百分比 (%BWB)。

    参数:
        tickers (List[str]): 股票代码列表。

    返回值:
        pd.DataFrame: 包含日期和所有股票的 SMA、下轨、上轨和 %BWB 的 DataFrame。
    """

    # 读取数据
    df = pd.read_csv("full_stock_data.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')  # 确保日期格式正确

    # 设置参数
    N = 20  # 时间周期
    K = 2   # 标准差倍数

    # 创建一个空的 DataFrame 来存储结果
    result_df = pd.DataFrame({'Date': df['Date']})

    # 遍历每个 ticker
    for ticker in tickers:
        ticker = ticker.lower()  # 转换为小写，以匹配文件名和列名

        # 确保列名存在
        close_col = f"{ticker}_Close"
        if close_col not in df.columns:
            print(f"警告: 列名 '{close_col}' 在 CSV 文件中不存在，跳过 {ticker}。")
            continue  # 跳过当前 ticker

        # 计算 SMA (中轨)
        result_df[f'{ticker}_SMA'] = df[close_col].rolling(window=N).mean()

        # 计算标准差
        result_df[f'{ticker}_StdDev'] = df[close_col].rolling(window=N).std()

        # 计算上轨和下轨
        result_df[f'{ticker}_UpperBand'] = result_df[f'{ticker}_SMA'] + (K * result_df[f'{ticker}_StdDev'])
        result_df[f'{ticker}_LowerBand'] = result_df[f'{ticker}_SMA'] - (K * result_df[f'{ticker}_StdDev'])

        # 计算 %BWB
        result_df[f'{ticker}_BWBp'] = (df[close_col] - result_df[f'{ticker}_LowerBand']) / (result_df[f'{ticker}_UpperBand'] - result_df[f'{ticker}_LowerBand']) * 100

        # 删除辅助列 'StdDev'
        result_df.drop(columns=[f'{ticker}_StdDev'], inplace=True)

    # result_df.to_csv("bwb.csv", index=False)

    # 转换为 long 形式 (改进方法)
    melted_dfs = []  # 用于存储每个指标的 melt 结果
    for metric in ['SMA', 'StdDev', 'UpperBand', 'LowerBand', 'BWBp']:
        temp_df = result_df.melt(
            id_vars=['Date'],
            value_vars=[col for col in result_df.columns if col.endswith(f"_{metric}")],
            var_name='Ticker',
            value_name=metric  # 使用指标名作为列名
        )
        temp_df['Ticker'] = temp_df['Ticker'].str.replace(f"_{metric}", "")  # 提取 ticker
        melted_dfs.append(temp_df)

    # 合并 melted DataFrames (使用 reduce)
    from functools import reduce
    result_df_long = reduce(lambda left, right: pd.merge(left, right, on=['Date', 'Ticker'], how='outer'),
                            melted_dfs)

    # 重新排列
    result_df_long = result_df_long[['Date', 'Ticker', 'SMA', 'StdDev', 'UpperBand', 'LowerBand', 'BWBp']]
    # 按日期和股票代码排序
    result_df_long = result_df_long.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)

    result_df_long = result_df_long.drop('StdDev', axis=1)

    # 输出 long 形式
    # print(result_df_long.head(200))

    # 读取 long 形式的布林带数据
    result_df_long['Date'] = pd.to_datetime(result_df_long['Date'])

    # 读取 "new_30_data.csv"
    new_data_df = pd.read_csv("new_30_data.csv")
    new_data_df['Date'] = pd.to_datetime(new_data_df['Date'])

    # 确保 "new_30_data.csv" 包含 "Stocks" 列
    if 'Stocks' not in new_data_df.columns:
        raise ValueError("'new_30_data.csv' 文件必须包含 'Stocks' 列。")

    # 重命名 long_df 中的 'Ticker' 列为 'Stocks'，以便合并
    long_df = result_df_long.rename(columns={'Ticker': 'Stocks'})

    # 检查 Stocks 列的值是否都在 long_df 的 Ticker 列中
    missing_stocks = set(new_data_df['Stocks']) - set(long_df['Stocks'])
    if missing_stocks:
        print(f"警告：以下股票代码在布林带数据中缺失，将不会被添加 BWBp 值：{missing_stocks}")

    # 使用 merge 将 BWBp 数据添加到 new_data_df
    merged_df = pd.merge(
        new_data_df,
        long_df[['Date', 'Stocks', 'BWBp']],  # 只选择需要的列
        on=['Date', 'Stocks'],  # 基于 Date 和 Stocks 进行合并
        how='left'  # 左连接
    )

    # 将合并后的 DataFrame 覆盖写入 "new_30_data.csv"
    merged_df.to_csv("new_30_data.csv", index=False)
    print(f"BWBp saved to new_30_data.csv")


def price(tickers):
    df = pd.read_csv("full_stock_data.csv")
    new_var_df = pd.read_csv("new_30_data.csv")
    price_col_mapping = {ticker: f'{ticker}_Close' for ticker in tickers}

    new_var_df['Price'] = np.nan

    for index, row in new_var_df.iterrows():
        date = row['Date']
        stock = row['Stocks']

        volume_row = df[df['Date'] == date]
        if not volume_row.empty:
            volume_value = volume_row[price_col_mapping[stock]].iloc[0]
            new_var_df.loc[index, 'Price'] = volume_value

    new_var_df.to_csv("new_30_data.csv", index=False)
    print("Close Price saved to new_30_data.csv")


def news_mentions(tickers):
    # Read the CSV file using pandas
    df = pd.read_csv("data/news_mentions.csv")

    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    df_melted = df.melt(id_vars=['Date'], var_name='Stocks', value_name='NewsMention')

    df_melted['Stocks'] = df_melted['Stocks'].str.lower()
    df_melted = df_melted.sort_values(by='Date')

    new_var_df = pd.read_csv("new_30_data.csv")

    # Create a copy to avoid modifying the original DataFrame in place.
    new_var_df_modified = new_var_df.copy()

    # Merge new_var_df with df_melted on 'Date' and 'stocks'
    merged_df = new_var_df_modified.merge(df_melted, on=['Date', 'Stocks'], how='left')

    merged_df.to_csv("new_30_data.csv", index=False)
    print("News Mentions saved to new_30_data.csv")


def investor_attention(tickers):
    # Read the CSV file using pandas
    df = pd.read_csv("data/InvestorAttention.csv")

    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    df_melted = df.melt(id_vars=['Date'], var_name='Stocks', value_name='InvestorAttention')

    df_melted['Stocks'] = df_melted['Stocks'].str.lower()
    df_melted = df_melted.sort_values(by='Date')

    new_var_df = pd.read_csv("new_30_data.csv")

    # Create a copy to avoid modifying the original DataFrame in place.
    new_var_df_modified = new_var_df.copy()

    # Merge new_var_df with df_melted on 'Date' and 'stocks'
    merged_df = new_var_df_modified.merge(df_melted, on=['Date', 'Stocks'], how='left')

    merged_df.to_csv("new_30_data.csv", index=False)
    print("Investors Attention saved to new_30_data.csv")


def calculate_rsi(tickers):
    df = pd.read_csv("full_stock_data.csv")
    period = 14
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date'].drop(1)

    price_col_mapping = {ticker: f'{ticker}_Close' for ticker in tickers}
    for column_name in df.columns:
        if column_name in price_col_mapping.values(): #check the values, not the keys.
            delta = df[column_name].diff()
            delta = delta[1:]

            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0

            roll_up1 = up.rolling(window=period).mean()
            roll_down1 = abs(down.rolling(window=period).mean())

            rs = roll_up1 / roll_down1
            rsi = 100.0 - (100.0 / (1.0 + rs))

            # Corrected: Align indices and change column name
            new_df[column_name.replace("_Close", "")] = rsi.values #use .values to remove index.
    new_df = new_df.dropna()

    # Iterate through all columns except the 'Date' column
    for column_name in new_df.columns:
        if column_name != 'Date':
            # Apply the transformation logic
            new_df[column_name] = new_df[column_name].apply(lambda x: 1 if x < 30 or x > 70 else 0)

    df_long = new_df.melt(id_vars=['Date'], var_name='Stocks', value_name='RSI')

    df_long = df_long.sort_values(by='Date')

    new_var_df = pd.read_csv("new_30_data.csv")

    # Create a copy to avoid modifying the original DataFrame in place.
    new_var_df_modified = new_var_df.copy()

    # Merge new_var_df with df_melted on 'Date' and 'stocks'
    merged_df = new_var_df_modified.merge(df_long, on=['Date', 'Stocks'], how='left')

    merged_df.to_csv("new_30_data.csv", index=False)
    print("RSI saved to new_30_data.csv")


def csad(tickers):
    # 读取 CSSD 计算后的数据
    df = pd.read_csv("new_30_data.csv")

    # 重新读取原始股票数据以计算 CSAD
    stock_df = pd.read_csv('full_stock_data.csv')

    try:
        stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    except ValueError:
        stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.strftime('%Y-%m-%d')

    # 计算收益率
    for ticker in tickers:
        stock_df[f'{ticker}_Return'] = stock_df[f'{ticker}_Close'].pct_change()
    stock_df['sp500_Return'] = stock_df['sp500_Close'].pct_change()

    stock_df = filter_date(stock_df)  # 过滤日期

    # 计算 CSAD
    rm_column = 'sp500_Return'
    unique_dates = stock_df['Date'].unique()

    csad_data = []
    for date in unique_dates:
        for ticker in tickers:
            row = stock_df[stock_df['Date'] == date].iloc[0]
            rj_values = row[[f'{t}_Return' for t in tickers if t != ticker]].values
            rm_value = row[rm_column]

            abs_diff_sum = np.sum(np.abs(rj_values - rm_value))
            csad_value = abs_diff_sum / (len(tickers) - 2)

            csad_data.append({'Date': date, 'Stocks': ticker, 'CSAD': csad_value})

    csad_df = pd.DataFrame(csad_data)

    # 统一 Date 列的数据类型
    df['Date'] = pd.to_datetime(df['Date'])
    csad_df['Date'] = pd.to_datetime(csad_df['Date'])

    # 合并 CSSD 和 CSAD 数据
    merged_df = pd.merge(df, csad_df, on=['Date', 'Stocks'], how='left')
    max_value = merged_df['CSAD'].max() + 1
    merged_df['CSAD'] = max_value - merged_df['CSAD']

    # 保存最终结果
    merged_df.to_csv("new_30_data.csv", index=False)
    print("CSAD saved to new_30_data.csv")


# define tickers
tickers = ["msft", "googl", "amzn", "meta", "bidu", "nvda", "amd", "tsm", "crm", "adbe"]

cssd(tickers)
wr(tickers)
trading_volume(tickers)
turnover_ratio(tickers)
# rolling_volatility(tickers)
sentiment(tickers)
market_return()
rolling_volatility(tickers)
vix()
bwbp(tickers)
price(tickers)
news_mentions(tickers)
investor_attention(tickers)
calculate_rsi(tickers)
csad(tickers)
