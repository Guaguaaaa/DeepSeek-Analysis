import pandas as pd
import numpy as np
from itertools import combinations, product

# 定义一个函数来计算 HTMT

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
            print(f"Note: Latent variable '{lv1}' has only one indicator — skipping internal correlation.")
            mean_within1 = 1.0
        else:
            mean_within1 = np.mean(within1_corrs)

        if not within2_corrs:
            print(f"Note: Latent variable '{lv2}' has only one indicator — skipping internal correlation.")
            mean_within2 = 1.0
        else:
            mean_within2 = np.mean(within2_corrs)

        denominator = np.sqrt(mean_within1 * mean_within2)
        htmt = numerator / denominator if denominator != 0 else np.nan

        htmt_matrix.loc[lv1, lv2] = htmt
        htmt_matrix.loc[lv2, lv1] = htmt

    np.fill_diagonal(htmt_matrix.values, 1.0)
    return htmt_matrix

# 示例
if __name__ == "__main__":
    data = pd.read_csv("post_30_data.csv")
    latent_vars = {
        "Anchoring": ["Price", "BWBp"],
        "FoMO": ["Sentiment", "InvestorAttention"],
        "CSSD": ["CSSD"]
    }
    htmt_values = compute_htmt(data, latent_vars)
    print(htmt_values)
