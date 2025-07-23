import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from itertools import combinations, product
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.vectors import StrVector


def apply_sem(data_pls, outer_relation):
    pandas2ri.activate()
    robjects.r('.libPaths("/home/peter/R/x86_64-pc-linux-gnu-library/4.3")')
    robjects.r('options(warn=-1)')  # -1 表示不显示任何警告

    # 删除包含缺失值的行
    # data_pls = data_pls.dropna(subset=['Sentiment'])

    # 缩放数据
    data_pls_scaled = pd.DataFrame(scale(data_pls), columns=data_pls.columns)

    # print("Scaled Data Head:")
    # print(data_pls_scaled.head())

    # 计算相关矩阵
    cor_matrix = data_pls_scaled.corr()
    print("\nCorrelation Matrix of Scaled Data:")
    print(cor_matrix)

    # 激活rpy2环境
    robjects.r.source('./activateR.R')

    # 将 pandas DataFrame 转换为 R DataFrame
    r_data = pandas2ri.py2rpy(data_pls_scaled)
    robjects.globalenv["r_data"] = r_data

    # 定义 R 模型的内部模型和外部模型
    inner_model = robjects.r('''
      inner_model <- matrix(c(0, 0, 0, 0,
                              0, 0, 0, 0,
                              0, 0, 0, 0,
                              1, 1, 1, 0),
                            nrow = 4, ncol = 4, byrow = TRUE,
                            dimnames = list(c("Herding", "Overconfidence", "FoMO", "MarketInstability"),
                                            c("Herding", "Overconfidence", "FoMO", "MarketInstability")))
      inner_model
    ''')

    # 转换 outer_relation 到 R list（outer_model）
    outer_model_r = robjects.ListVector({key: StrVector(value) for key, value in outer_relation.items()})

    # 从 outer_relation 创建 blocks
    blocks = robjects.ListVector({key: StrVector(value) for key, value in outer_relation.items()})
    blocks.names = robjects.StrVector(list(outer_relation.keys()))

    modes = robjects.StrVector(["A", "A", "A", "A"])
    robjects.r.assign("modes", modes)
    robjects.r('''names(modes) <- c("Herding", "Overconfidence", "FoMO", "MarketInstability")''')

    # 加载 R 包
    plspm = importr('plspm')
    robjects.r('set.seed(123)')

    # 检查NA
    na_check = robjects.r('any(is.na(r_data))')[0]

    if na_check:
        print("NAs found in data_pls_scaled. Removing rows with NAs in R.")
        robjects.r('r_data <- na.omit(r_data)')


    # print(robjects.r('ls()'))  # 查看 R 环境中的变量
    # robjects.r('print(head(r_data))')  # 确保 r_data 存在


    # 运行 PLS-SEM 分析
    try:
        pls_model = plspm.plspm(r_data, inner_model, blocks, robjects.r['modes'], boot_val=True, br=5000)
        robjects.globalenv["pls_model"] = pls_model  # 确保它存入 R 环境
    except Exception as e:
        print("Error in plspm.plspm():", e)


    print("\n-----------Summaries-----------")
    summary_results = robjects.r('summary(pls_model)')
    print(summary_results)

    # Extract latent variable scores
    lv_scores = r('pls_model$scores')

    n_latent_vars = lv_scores.shape[1]
    column_names = [f"LV{i + 1}" for i in range(n_latent_vars)]

    lv_scores_df = pd.DataFrame(lv_scores, columns=column_names)

    # print(lv_scores_df.head())


    pls_model = robjects.r["pls_model"]  # 先获取 pls_model

    cr_r = pls_model.rx2("unidim")
    cr = pandas2ri.rpy2py(cr_r)
    #print(cr)

    p_value_r = pls_model.rx2("inner_model")
    p_value = p_value_r.rx2("LV4")
    # print(p_value_r)
    # print(p_value)

    ave_r = pls_model.rx2("inner_summary")
    ave = pandas2ri.rpy2py(ave_r)
    #print(ave)


    boot_r = pls_model.rx2("boot").rx2("paths")
    boot = pandas2ri.rpy2py(boot_r)
    # print(boot)

    loadings_r = pls_model.rx2("boot").rx2("loadings")
    outer_loadings = pandas2ri.rpy2py(loadings_r)
    loading_value = [outer_loadings.loc["LV4-TradingVolume", "Original"],
                     outer_loadings.loc["LV4-TurnoverRatio", "Original"]]
    # print(loading_value)


    result = pd.DataFrame()
    result['CR'] = cr['DG.rho']
    values = [p_value[1, 3], p_value[2, 3], p_value[3, 3]]
    result['P_Value'] = values + [np.nan]
    coef_value = [p_value[1, 0], p_value[2, 0], p_value[3, 0]]
    result['Coefficient'] = coef_value + [np.nan]
    result['AVE'] = ave['AVE']
    result['perc025'] = list(boot['perc.025']) + [np.nan]
    result['perc975'] = list(boot['perc.975']) + [np.nan]
    result.index = ['Herding', 'Overconfidence', 'FoMO', 'Market Instability']
    # print(result)
    return result, lv_scores_df, loading_value


def compute_htmt(data, latent_vars):
    htmt_matrix = pd.DataFrame(index=latent_vars.keys(), columns=latent_vars.keys(), dtype=float)

    for lv1, lv2 in combinations(latent_vars.keys(), 2):
        indicators1 = latent_vars[lv1]
        indicators2 = latent_vars[lv2]

        # 跨构面相关（numerator）
        cross_corrs = [abs(data[i].corr(data[j])) for i, j in product(indicators1, indicators2)]
        cross_corrs = [x for x in cross_corrs if not pd.isna(x)]
        numerator = np.mean(cross_corrs) if cross_corrs else np.nan

        # 同构面相关（denominator）
        within1_corrs = [abs(data[i].corr(data[j])) for i, j in combinations(indicators1, 2)]
        within2_corrs = [abs(data[i].corr(data[j])) for i, j in combinations(indicators2, 2)]

        within1_corrs = [x for x in within1_corrs if not pd.isna(x)]
        within2_corrs = [x for x in within2_corrs if not pd.isna(x)]

        if not within1_corrs:
            # print(f"Note: Latent variable '{lv1}' has only one indicator — skipping internal correlation.")
            mean_within1 = 1.0
        else:
            mean_within1 = np.mean(within1_corrs)

        if not within2_corrs:
            # print(f"Note: Latent variable '{lv2}' has only one indicator — skipping internal correlation.")
            mean_within2 = 1.0
        else:
            mean_within2 = np.mean(within2_corrs)

        denominator = np.sqrt(mean_within1 * mean_within2)
        htmt = numerator / denominator if denominator != 0 else np.nan

        htmt_matrix.loc[lv1, lv2] = htmt
        htmt_matrix.loc[lv2, lv1] = htmt

    np.fill_diagonal(htmt_matrix.values, 1.0)
    return htmt_matrix


def compute_vif(df):
    X = sm.add_constant(df)
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i + 1) for i in range(len(df.columns))]  # +1 to skip constant
    return vif_data


# Function to calculate predicted latent scores (based on path model)
def predict_latent_scores(data, path_coefs):
    predicted_latent = pd.DataFrame(index=data.index)
    for latent, coefs in path_coefs.items():
        predicted_latent[latent] = 0
        for indicator, coef in coefs.items():
            predicted_latent[latent] += data[indicator] * coef
    return predicted_latent


# Function to calculate Q² using blindfolding (K-fold cross-validation)
def calculate_q2(X_indicators, Y_endogenous, latent_scores, path_coefficients, n_folds=10):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    q2_values = {}

    for train_index, test_index in kf.split(X_indicators):
        X_train, X_test = X_indicators.iloc[train_index], X_indicators.iloc[test_index]
        Y_train, Y_test = Y_endogenous.iloc[train_index], Y_endogenous.iloc[test_index]
        latent_train = latent_scores.iloc[train_index]
        latent_test = latent_scores.iloc[test_index]

        # Predict latent scores based on the training data's path model
        predicted_latent_train = predict_latent_scores(X_train, path_coefficients)

        # Train a regression model to predict endogenous variables
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(pd.concat([X_train, predicted_latent_train, latent_train], axis=1), Y_train)

        # Predict endogenous variables on the test set
        predicted_endogenous = model.predict(pd.concat([X_test, predict_latent_scores(X_test, path_coefficients), latent_test], axis=1))

        # Calculate Q² for each endogenous variable
        for col in Y_endogenous.columns:
            rss = np.sum((Y_test[col].values - predicted_endogenous[:, Y_endogenous.columns.get_loc(col)]) ** 2)
            tss = np.sum((Y_test[col].values - np.mean(Y_test[col].values)) ** 2)
            q2 = 1 - (rss / tss)
            if col not in q2_values:
                q2_values[col] = []
            q2_values[col].append(q2)

    # Average Q² values across folds
    average_q2 = {col: np.mean(values) for col, values in q2_values.items()}
    return average_q2


def calculate_weighted_q2(q2_values, loadings):
    """
    Calculates a weighted average of Q² values based on loadings.

    Args:
        q2_values (dict): A dictionary where keys are indicator names and values are Q² values.
        loadings (dict): A dictionary where keys are indicator names and values are loadings.

    Returns:
        float: The weighted average Q².
    """

    # Ensure that both dictionaries have the same keys
    if q2_values.keys() != loadings.keys():
        raise ValueError("Q² values and loadings dictionaries must have the same keys.")

    # Calculate weighted Q²
    weighted_sum = 0
    total_weight = 0

    for indicator, q2 in q2_values.items():
        loading = loadings[indicator]
        weighted_sum += q2 * loading
        total_weight += loading

    if total_weight == 0:
        return 0  # Avoid division by zero

    weighted_q2 = weighted_sum / total_weight
    return weighted_q2


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.3f}'.format)
    pd.set_option('display.max_colwidth', None)

    # The following line loads the confidential dataset, which is not included in this repository.
    # Therefore, this line has been commented out.
    # data = pd.read_csv("data/example_data.csv")

    # 选择相关列
    data_pls = data[['CSSD', 'TradingVolume', 'TurnoverRatio', 'Sentiment', 'BWBp',
                     'Price', 'InvestorAttention']].copy()

    model_1 = {
        "Herding": ['CSSD'],
        "Overconfidence": ['BWBp', 'Price'],
        "FoMO": ['Sentiment', 'InvestorAttention'],
        "MarketInstability": ['TradingVolume', 'TurnoverRatio']
    }

    print("\n-----------Core Result-----------")
    result, lv_scores_df, outer_loadings = apply_sem(data_pls, model_1)
    print("Core factors:")
    print(result)

    htmt_values = compute_htmt(data, model_1)
    print("\n-----------HTMT Matrix-----------")
    print(htmt_values)

    # 将所有测量变量合并后计算 VIF
    all_indicators = sum(model_1.values(), [])
    vif_result = compute_vif(data[all_indicators])
    print("\n-----------VIF-----------")
    print(vif_result)

    # q^2
    # Load the data
    data_test_model = data.copy(deep=True)

    data_test_model = data_test_model.drop(columns=['Date', 'Stocks'])

    # Scale the original data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_test_model)
    data_scaled_df = pd.DataFrame(data_scaled, columns=data_test_model.columns)

    # Prepare the data
    X_indicators = data_scaled_df[['CSSD', 'BWBp', 'Price', 'Sentiment', 'InvestorAttention']]
    Y_endogenous = data_scaled_df[['TradingVolume', 'TurnoverRatio']]

    # Add latent scores to X
    X = pd.concat([X_indicators, lv_scores_df], axis=1)

    # Path model equations
    lv1_coef = result.loc["Herding", "Coefficient"]
    lv2_coef = result.loc["Overconfidence", "Coefficient"]
    lv3_coef = result.loc["FoMO", "Coefficient"]
    path_coefficients = {
        'LV1': {'CSSD': lv1_coef},
        'LV2': {'BWBp': lv2_coef, 'Price': lv2_coef},
        'LV3': {'Sentiment': lv3_coef, 'InvestorAttention': lv3_coef},
        # MarketInstability is already in latent scores, so no path needed
    }

    # Calculate and print Q² values
    q2_results = calculate_q2(X_indicators, Y_endogenous, lv_scores_df, path_coefficients)
    loadings = {'TradingVolume': outer_loadings[0], 'TurnoverRatio': outer_loadings[1]}
    print("\n-----------Q-sq-----------")
    print("Q² values:", q2_results)

    weighted_q2_overconfidence = calculate_weighted_q2(q2_results, loadings)
    print(f"Weighted Q² for Overconfidence: {weighted_q2_overconfidence}")

