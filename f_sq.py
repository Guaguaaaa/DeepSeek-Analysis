# f²计算函数
def calculate_f_squared(r2_included, r2_excluded):
    if r2_included == 1:
        raise ValueError("R² included cannot be 1 (would cause division by zero).")
    f2 = (r2_included - r2_excluded) / (1 - r2_included)
    return f2

# 示例输入：假设去除 Overconfidence 变量
# for privacy reason real r-sq value will not be presented
r2_included = -1  # 原始模型中的 R²
r2_excluded = -1  # 去掉某个外生变量后的 R²

# 计算 f²
f2_result = calculate_f_squared(r2_included, r2_excluded)

# 输出结果并解释效应大小
print(f"f² 值为: {f2_result:.4f}")

# 效应大小判断
if f2_result < 0.02:
    size = "negligible"
elif f2_result < 0.15:
    size = "small"
elif f2_result < 0.35:
    size = "medium"
else:
    size = "large"

print(f"根据 Cohen 的标准，该效应为: {size}")
