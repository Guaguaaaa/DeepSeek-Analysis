import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.rinterface_lib.sexp
from rpy2.robjects.vectors import StrVector

import copy
from tqdm import tqdm
import json
import psutil
import time


def apply_sem(data_pls, outer_relation):
    pandas2ri.activate()
    robjects.r('.libPaths("/home/peter/R/x86_64-pc-linux-gnu-library/4.3")')
    robjects.r('options(warn=-1)')  # -1 表示不显示任何警告

    # 删除包含缺失值的行
    data_pls = data_pls.dropna(subset=['Sentiment'])

    # 缩放数据
    data_pls_scaled = pd.DataFrame(scale(data_pls), columns=data_pls.columns)

    # print("Scaled Data Head:")
    # print(data_pls_scaled.head())

    # 计算相关矩阵
    cor_matrix = data_pls_scaled.corr()
    # print("\nCorrelation Matrix of Scaled Data:")
    # print(cor_matrix)

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
        pls_model = plspm.plspm(r_data, inner_model, blocks, robjects.r['modes'], boot_val=True, br=200)
        robjects.globalenv["pls_model"] = pls_model  # 确保它存入 R 环境
    except Exception as e:
        print("Error in plspm.plspm():", e)


    # print("\n-----------Summaries-----------")
    summary_results = robjects.r('summary(pls_model)')
    # print(summary_results)

    pls_model = robjects.r["pls_model"]  # 先获取 pls_model

    cr_r = pls_model.rx2("unidim")
    cr = pandas2ri.rpy2py(cr_r)
    #print(cr)

    p_value_r = pls_model.rx2("inner_model")
    p_value = p_value_r.rx2("LV4")
    #print(p_value_r)
    #print(p_value)

    ave_r = pls_model.rx2("inner_summary")
    ave = pandas2ri.rpy2py(ave_r)
    #print(ave)


    boot_r = pls_model.rx2("boot").rx2("paths")
    boot = pandas2ri.rpy2py(boot_r)
    # print(boot)


    result = pd.DataFrame()
    result['CR'] = cr['DG.rho']
    values = [p_value[1, 3], p_value[2, 3], p_value[3, 3]]
    result['P_Value'] = values + [np.nan]
    result['AVE'] = ave['AVE']
    result['perc025'] = list(boot['perc.025']) + [np.nan]
    result['perc975'] = list(boot['perc.975']) + [np.nan]
    result.index = ['Herding', 'Overconfidence', 'FoMO', 'Market Instability']
    # print(result)
    return result


"""
loadings_r = pls_model.rx2("outer_model")
loadings_r = robjects.r('''
    outer_model <- pls_model$outer_model
    outer_model[] <- lapply(outer_model, function(x) if(is.factor(x)) as.character(x) else x)
    outer_model
''')
df = pandas2ri.rpy2py(loadings_r)
# print(df)
"""

"""
# 提取不同部分
pls_model = robjects.r["pls_model"]  # 先获取 pls_model

# summary 中结果变量名对应关系
pls_results = {
    "blocks_unidimensionality": pls_model.rx2("unidim"),
    "outer_model": pls_model.rx2("outer_model"),
    "crossloadings": pls_model.rx2("crossloadings"),
    "inner_model": pls_model.rx2("inner_model"),
    "model": pls_model.rx2("model"),
    "summary_inner_model": pls_model.rx2("inner_summary"),
    "goodness_of_fit": pls_model.rx2("gof"),
    "total_effects": pls_model.rx2("effects"),
    "bootstrap_validation": pls_model.rx2("boot"),
}
"""

def generate_combinations(current_dict, remaining_elements, results):
    if not remaining_elements:
        # 检查每个值列表是否至少有两个元素
        valid = True
        for values in current_dict.values():
            if len(values) < 2:
                valid = False
                break
        if valid:
            results.append(copy.deepcopy(current_dict))  # 使用深拷贝
        return

    element = remaining_elements[0]
    available_keys = element_availability[element]

    for key in available_keys:
        if key in current_dict:
            new_dict = copy.deepcopy(current_dict) #每次循环都必须深度拷贝，确保new_dict的初始值不会在下次循环时改变。
            new_dict[key].append(element)
            generate_combinations(new_dict, remaining_elements[1:], results)
    # 如果data中的某一个值，没有插入outer_relation中，也应该保存一份，所以，此处也需要回溯。
    generate_combinations(current_dict, remaining_elements[1:], results)


def test_model(result):
    cr_score = 0
    ave_score = 0
    p_score = 0

    for index, value in result['CR'].items():
        if value < 0.7:
            return False, -1
        else:
            cr_score += (value - 0.7) / 0.3
    for index, value in result['AVE'].items():
        if value < 0.5:
            return False, -1
        else:
            ave_score += (value - 0.5) / 0.5
    for index, value in result['P_Value'].items():
        if not np.isnan(value) and value > 0.05:
            return False, -1
        else: p_score += (0.05 - value) / 0.05

    achv_score = (cr_score * 0.4) + (ave_score * 0.4) + (p_score * 0.2)
    return True, achv_score


if __name__ == "__main__":
    # 加载数据
    data = pd.read_csv("new_30_data.csv")

    # 选择相关列
    data_pls = data[['CSSD', 'WR', 'TradingVolume', 'TurnoverRatio', 'Sentiment', 'RollingVolatility', 'VIX', 'BWBp',
                     'Price', 'MarketReturn', 'NewsMention', 'InvestorAttention', 'RSI']].copy()

    col_names = ['CSSD', 'WR', 'TradingVolume', 'TurnoverRatio', 'Sentiment', 'RollingVolatility', 'BWBp', 'VIX',
                 'MarketReturn', 'Price', 'NewsMention', 'InvestorAttention', 'RSI']


    outer_relation = {
        "Herding": [],
        "Overconfidence": [],
        "FoMO": [],
        "MarketInstability": []
    }

    element_availability = {
        'CSSD': ['Herding'],
        'WR': ['Herding'],
        'TradingVolume': ['Overconfidence', 'FoMO', 'MarketInstability'],
        'TurnoverRatio': ['Overconfidence', 'MarketInstability'],
        'Sentiment': ['FoMO'],
        'RollingVolatility': ['Overconfidence', 'MarketInstability'],
        'BWBp': ['MarketInstability', "Overconfidence"],
        'VIX': ['FoMO', "MarketInstability"],
        "MarketReturn": ["MarketInstability"],
        'Price': ['MarketReturn', 'Overconfidence', 'FoMO'],
        'NewsMention': ['FoMO'],
        'InvestorAttention': ['FoMO'],
        'RSI': ['Herding']
    }

    combinations = []
    initial_dict = copy.deepcopy(outer_relation)
    generate_combinations(initial_dict, col_names, combinations)
    """for combination in combinations:
        print(combination)
    print(f"Total Combinations: {len(combinations)}")"""

    keep = []
    failed = []
    scores = []

    """process = psutil.Process()
    process.cpu_affinity([0, 1, 2, 3, 4, 5, 6, 7])"""

    start_index = 11100
    sliced_combinations = combinations[start_index:]

    for model in tqdm(sliced_combinations):
        result = apply_sem(data_pls, model)
        res_bool, score = test_model(result)
        if res_bool:
            keep.append(model)
            temp = copy.deepcopy(model)
            temp['score'] = score
            scores.append(temp)
            print(scores)
        else:
            failed.append(model)
            if score != -1:
                print(f"Error analysed result: Failed model returns a score of {score}")
                print(f"Model:\n{model}")

        # Dynamic time.sleep based on CPU usage
        """cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > 85:  # Adjust threshold as needed
            time.sleep(1)  # Longer sleep during high CPU usage
        else:
            time.sleep(0.1)  # Shorter sleep otherwise"""

    with open('Achieved.json', 'w') as f:
        json.dump(keep, f, indent=4)
    with open('Failed.json', 'w') as f:
        json.dump(failed, f, indent=4)
    with open('scores.json', 'w') as f:
        json.dump(scores, f, indent=4)

    print(f'----------Complete----------')
    print(f'Total Achieved Models: {len(keep)}')
    userInput = input("Show all models? (Y/n)")
    if userInput == "Y":
        print(keep)


    """
    Example model path list:
    --------------------------------------------
    my_list = [
        {
            'key_1': ['element_1', 'element_2'],
            'key_2': ['element_3', 'element_4'],
            'key_3': ['element_5', 'element_6'],
            'key_4': ['element_7', 'element_8']
        },
        {
            'key_1': ['element_9', 'element_10'],
            'key_2': ['element_11', 'element_12'],
            'key_3': ['element_13', 'element_14'],
            'key_4': ['element_15', 'element_16']
        }
    ]
    """


    """
    Pseudo Code:
    -------------------------------
    加载data, data_pls
    设置 col_names
    找到所有的 outer_relation 的排列组合，并存入 combinations 中。
    初始化 list(Dictionary) 类型变量 keep，用于存储合格的 indicator 组合
    for model in combinations:
        result = apply_sem(data_pls, model)
        检测返回的 result 结果
        if 结果合适 then 
            keep.append(model)
        else
            尝试下一个 model 组合
    print(keep)
    print(f"Total achieved: {len(keep)}")
    """