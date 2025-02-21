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

# 生成分组序号
quota['group_sep'] = quota.groupby('group').cumcount() + 1

# 筛选数据
if use_sample_cons['condition'].lower().strip() == 't':
    data = all_data.copy()
else:
    condition = convert_condition(use_sample_cons['condition'])
    data = all_data.query(condition).copy()

# 准备转换矩阵（修复初始化问题）
categories = [str(i) for i in range(1, 7)] + ['missing']
trans = pd.DataFrame({
    'userID': data['userID'].astype(str),
    'target1': pd.Categorical(np.full(len(data), 'missing'), categories=categories)
})

# 填充目标列（确保覆盖所有样本）
for _, row in quota.iterrows():
    condition = convert_condition(row['condition'])
    mask = data.eval(condition)
    trans.loc[mask, 'target1'] = str(row['group_sep'])

# 处理未覆盖样本（关键修复）
print(f"未覆盖样本数: {len(trans[trans['target1'] == 'missing'])}")
trans = trans[trans['target1'] != 'missing'].reset_index(drop=True)

# 构造约束条件（完全匹配分类）
group_data = quota[quota['group'] == 1]
total_target = group_data['target'].sum()

constraint_dict = {
    str(i): group_data[group_data['group_sep'] == i]['target'].values[0] / total_target
    for i in range(1, 7)
}
constraint_series = pd.Series(
    data=[constraint_dict.get(cat, 0) for cat in categories if cat != 'missing'],
    index=pd.CategoricalIndex([str(i) for i in range(1, 7)], name='target1')
)

# 验证约束条件
print("\n约束条件验证:")
print(constraint_series)
print(f"总比例: {constraint_series.sum():.2%}")

# IPFN配置（最终正确参数）
trans['weight'] = 1.0

ipf = ipfn.ipfn(
    trans,
    aggregates=[constraint_series.values],
    dimensions=[[['target1']]],
    weight_col='weight',
    max_iteration=5000,
    verbose=2
)

# 执行迭代
try:
    result = ipf.iteration()
    print("\n加权成功！")

    # 计算最终权重
    wt = result['weight'] * (use_sample_cons['target'] / len(all_data))
    data = data.iloc[trans.index].copy()  # 保持索引一致
    data['weight'] = wt.values
    data.to_csv("output.csv", index=False)

    # 计算有效样本量
    n_eff = (data['weight'].sum() ** 2) / (data['weight'] ** 2).sum()
    print(f"有效样本量: {n_eff:.1f}")

except Exception as e:
    print(f"错误信息: {str(e)}")
    print("最终验证:")
    print("转换矩阵分类:", trans['target1'].unique())
    print("约束条件索引:", constraint_series.index.tolist())
