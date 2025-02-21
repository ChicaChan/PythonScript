import pandas as pd
import numpy as np
from ipfn import ipfn


def convert_condition(condition):
    """处理R风格的条件表达式"""
    return (
        condition.replace("data$", "")
        .replace("&", " and ")
        .replace("|", " or ")
        .replace("==", " == ")
        .replace("!=", " != ")
        .strip()
    )


# 读取数据
all_data = pd.read_excel("input.xlsx", sheet_name=0)
all_quota = pd.read_excel("input.xlsx", sheet_name=1)

# 处理样本条件
use_sample_cons = all_quota[all_quota['group'] == 99].iloc[0]
quota = all_quota[all_quota['group'] != 99].copy()

# 生成分组序号（关键修复）
quota['group_sep'] = quota.groupby('group').cumcount() + 1

# 筛选数据
if use_sample_cons['condition'].lower().strip() == 't':
    data = all_data.copy()
else:
    condition = convert_condition(use_sample_cons['condition'])
    data = all_data.query(condition).copy()

# 准备转换矩阵（修复total问题）
trans = pd.DataFrame({'userID': data['userID'].astype(str)})
trans['target1'] = pd.Series(dtype='string')

# 生成分类标签映射表
category_map = {i: str(i) for i in range(1, 7)}

# 填充目标列（确保无缺失）
for _, row in quota.iterrows():
    condition = convert_condition(row['condition'])
    mask = data.eval(condition)
    trans.loc[mask, 'target1'] = category_map[row['group_sep']]

# 处理缺失值（关键修复）
trans['target1'] = trans['target1'].fillna('missing').astype('category')

# 构造约束条件（精确匹配）
group_data = quota[quota['group'] == 1]
total_target = group_data['target'].sum()  # 重命名变量避免冲突

cons_dict = {
    'target1': {
        category_map[row['group_sep']]: row['target'] / total_target
        for _, row in group_data.iterrows()
    }
}

# 验证约束条件
print("验证约束条件:")
for k, v in cons_dict['target1'].items():
    print(f"  {k}: {v:.2%}")
print(f"总比例: {sum(cons_dict['target1'].values()):.2%}")

# Prepare weights for IPFN
trans['_weight_'] = 1.0  # Add initial weights column

# Modified IPFN configuration with 'observed' parameter set to False
ipf = ipfn.ipfn(
    trans,
    aggregates=[cons_dict['target1']],
    dimensions=[['target1']],
    weight_col='_weight_',  # Specify weight column
    max_iteration=5000,
    verbose=2,
    observed=False  # Add this parameter
)
# Execute iteration with error handling
try:
    weights = ipf.iteration()
    print("\n加权成功！")

    # Calculate final weights (modified)
    scale_factor = use_sample_cons['target'] / len(data)
    data['weight'] = weights * scale_factor

    # Save results
    data.to_csv("output.csv", index=False)

    # Calculate effective sample size
    n_eff = (data['weight'].sum() ** 2) / (data['weight'] ** 2).sum()
    print(f"有效样本量: {n_eff:.1f}")

except Exception as e:
    print(f"错误信息: {str(e)}")
    print("\n诊断信息:")
    print(f"转换矩阵大小: {trans.shape}")
    print("\n目标1分布:")
    print(trans['target1'].value_counts())
