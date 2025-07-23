import pandas as pd
import statsmodels.api as sm

# 加载整合后的数据
# The following line loads the confidential dataset, which is not included in this repository.
# Therefore, this line has been commented out.
# df_sum = pd.read_csv("data/example_data.csv", index_col='Date', parse_dates=True)
'''
start_date_filter = '2024-08-12'
end_date_filter = '2025-01-06'
df_sum = df_sum.loc[start_date_filter:end_date_filter]
print(f"数据已筛选至日期范围: {start_date_filter} - {end_date_filter}")
'''

# 定义自变量 (X) 和因变量 (y) 的列名
dependent_variable = 'HerdingIndicator_t'
independent_variables = ['OverConfidencet', 'FOMO_t', 'MarketReturn_t', 'Volatility_t']

# 准备自变量 (X) 和因变量 (y) 数据
y = df_sum[dependent_variable] # 因变量 (Herding Indicator)
X = df_sum[independent_variables] # 自变量 (OverConfidence, FOMO, MarketReturn, Volatility)

# 打印准备好的数据
print("因变量 (y) - HerdingIndicator_t:")
print(y.head())
print("\n自变量 (X):")
print(X.head())


# 添加常数项 (截距项) 到自变量矩阵 X
X = sm.add_constant(X) # statsmodels 的 Logit 模型需要手动添加常数项

# 拟合逻辑回归模型
logistic_model = sm.Logit(y, X) # 使用 Logit 函数创建逻辑回归模型
logistic_results = logistic_model.fit() # 拟合模型

# 7. 打印回归结果摘要
print(logistic_results.summary())