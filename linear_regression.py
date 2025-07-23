import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf


# 定义股票代码列表和日期范围
tickers = ["MSFT", "GOOGL", "AMZN", "META", "BIDU", "NVDA", "AMD", "TSM", "CRM", "ADBE"]
start_date = "2024-08-12"
end_date = "2025-02-07"

# 加载 CSV 文件
try:
    turnover_ratios_data = pd.read_csv("data/turnover_ratios.csv", parse_dates=['Date'])
    turnover_ratios_data['Date'] = pd.to_datetime(turnover_ratios_data['Date'], utc=True).dt.tz_localize(
        None).dt.date  # 转换 Date 列为只有日期的格式, 并去除时区信息
    turnover_ratios_data = turnover_ratios_data.set_index('Date')  # 将 Date 列设置为 index

    rolling_volatility_data = pd.read_csv("data/rolling_volatility.csv", index_col="Date", parse_dates=True)
    sp500_market_return_data = pd.read_csv("data/SP500_MarketReturn.csv", index_col="Date", parse_dates=True)
    fomo_sentiment_data = pd.read_csv("data/FOMO sentiment.csv", index_col="Date", parse_dates=True)
    cssd_data = pd.read_csv("data/Filtered_Stock_Data_with_CSSD.csv", index_col="Date", parse_dates=True)
    print("CSV 文件加载成功.")
except FileNotFoundError as e:
    print(f"文件加载失败，请检查文件路径是否正确: {e}")
    exit()


# 初始化新的 DataFrame 用于存储整合后的数据
target_date_index = sp500_market_return_data.index
turnover_ratios_data = turnover_ratios_data.reindex(target_date_index)
rolling_volatility_data = rolling_volatility_data.reindex(target_date_index)
fomo_sentiment_data = fomo_sentiment_data.reindex(target_date_index)
cssd_data = cssd_data.reindex(target_date_index)

df_sum = pd.DataFrame(index=sp500_market_return_data.index) # 使用原始 DataFrame 的日期索引
df_sum.index.name = 'Date'

# 计算 OverConfidencet (Turnover Ratio) 的每日平均值
overconfidence_cols = tickers # turnover_ratios.csv 的列名直接是股票代码
df_sum['OverConfidencet'] = turnover_ratios_data[overconfidence_cols].mean(axis=1)
#df_sum['Overconfidencet'] = turnover_ratios_data[overconfidence_cols]
#print("OverConfidencet 平均值计算完成.")

# 计算 Volatility_t 的每日平均值
volatility_cols = tickers # rolling_volatility.csv.csv 的列名直接是股票代码
df_sum['Volatility_t'] = rolling_volatility_data[volatility_cols].mean(axis=1)
#print("Volatility_t 平均值计算完成.")

# 获取 FOMO_t (Sentiment) 的每日平均值
fomo_cols = tickers # FOMO sentiment.csv 的列名直接是股票代码
df_sum['FOMO_t'] = fomo_sentiment_data[fomo_cols].mean(axis=1)
#print("FOMO_t 平均值计算完成.")

# 直接复制 MarketReturn_t (S&P500 Market Return)
df_sum['MarketReturn_t'] = sp500_market_return_data['MarketReturn']
#print("MarketReturn_t 数据复制完成.")

df_sum['cssd_t'] = cssd_data['CSSD']
#print("cssd_t 数据复制完成")

# 移除所有包含 NaN 的行
df_sum = df_sum.dropna()
#print("移除 NaN 值完成.")

# 打印 DataFrame 的 head
print("\n整合后的 DataFrame 的 head:")
print(df_sum)

#-----------------------------------------------------------------------------------------------------------------------
# 计算Linear Regression
#-----------------------------------------------------------------------------------------------------------------------

def linear_regress(df_sum):
    # 准备数据
    X = df_sum[['OverConfidencet', 'FOMO_t']]  # 自变量和控制变量
    y = df_sum['cssd_t']  # 因变量

    # 添加常数项
    X = sm.add_constant(X)

    # 构建线性回归模型
    model = sm.OLS(y, X)

    # 拟合模型
    results = model.fit()

    # 打印回归结果
    print(results.summary())


def correlation_Coefficient(df_sum):
    # 计算相关系数矩阵
    correlation_matrix = df_sum[['OverConfidencet', 'FOMO_t']].corr()

    # 提取 OverConfidence_t 和 FOMO_t 之间的相关系数
    correlation_coefficient = correlation_matrix.loc['OverConfidencet', 'FOMO_t']

    print(f"OverConfidence_t 和 FOMO_t 的相关系数: {correlation_coefficient}")


def vif(df_sum):

    # 准备自变量和因变量数据
    X = df_sum[['FOMO_t']]  # FOMO_t 作为自变量
    y = df_sum['OverConfidencet']  # OverConfidence_t 作为因变量
    X = sm.add_constant(X)  # 添加常数项

    # 构建辅助回归模型
    aux_model = sm.OLS(y, X)
    aux_results = aux_model.fit()

    # 计算 OverConfidence_t 的 VIF
    vif_overconfidence = variance_inflation_factor(X.values, 1)  # 自变量索引为 1 (FOMO_t)

    # 计算 FOMO_t 的 VIF
    X_fomo = df_sum[['OverConfidencet']]  # OverConfidence_t 作为自变量
    y_fomo = df_sum['FOMO_t']  # FOMO_t 作为因变量
    X_fomo = sm.add_constant(X_fomo)  # 添加常数项
    aux_model_fomo = sm.OLS(y_fomo, X_fomo)
    aux_results_fomo = aux_model_fomo.fit()
    vif_fomo = variance_inflation_factor(X_fomo.values, 1)  # 自变量索引为 1 (OverConfidence_t)

    print(f"OverConfidence_t 的 VIF 值: {vif_overconfidence}")
    print(f"FOMO_t 的 VIF 值: {vif_fomo}")


def standardization(df_sum):
    scaler = StandardScaler()
    df_sum['OverConfidence_t_scaled'] = scaler.fit_transform(df_sum[['OverConfidencet']])
    df_sum['FOMO_t_scaled'] = scaler.fit_transform(df_sum[['FOMO_t']])

    # 使用标准化后的变量进行回归
    X_scaled = df_sum[['OverConfidence_t_scaled', 'FOMO_t_scaled']]
    X_scaled = sm.add_constant(X_scaled)
    model_scaled = sm.OLS(df_sum['cssd_t'], X_scaled)
    results_scaled = model_scaled.fit()
    print(results_scaled.summary())
    print(f"Condition Number (Scaled Model): {np.linalg.cond(X_scaled)}")  # 重新计算 Condition Number

    formula = 'cssd_t ~ OverConfidence_t_scaled + FOMO_t_scaled + OverConfidence_t_scaled * FOMO_t_scaled'  # 加入交互项
    model_interaction = smf.ols(formula, data=df_sum)  # 使用 formula API
    results_interaction = model_interaction.fit()
    print(results_interaction.summary())


def centering(df_sum):
    df_sum['OverConfidence_t_centered'] = df_sum['OverConfidencet'] - df_sum[
        'OverConfidencet'].mean()
    df_sum['FOMO_t_centered'] = df_sum['FOMO_t'] - df_sum[
        'FOMO_t'].mean()

    # 使用中心化后的变量进行回归
    X_centered = df_sum[['OverConfidence_t_centered', 'FOMO_t_centered']]
    X_centered = sm.add_constant(X_centered)
    model_centered = sm.OLS(df_sum['cssd_t'], X_centered)
    results_centered = model_centered.fit()
    print(results_centered.summary())
    print(f"Condition Number (Centered Model): {np.linalg.cond(X_centered)}")  # 重新计算 Condition Number

# linear_regress(df_sum)
# correlation_Coefficient(df_sum)
# vif(df_sum)
standardization(df_sum)
# centering(df_sum)