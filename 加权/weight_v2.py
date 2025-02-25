import pandas as pd
import numpy as np
from balance import Sample
import re


def convert_condition(cond):
    """处理R风格条件到Python查询条件"""
    cond = re.sub(r'data\$(\w+)', r'\1', cond)
    cond = cond.replace("&", " and ").replace("|", " or ")
    cond = re.sub(r'(\w+)\s*==\s*(\d+)', r'\1 == \2', cond)
    cond = re.sub(r"(\w+)\s*==\s*'([^']+)'", r"\1 == '\2'", cond)
    return cond


# 读取数据并转换ID类型
data = pd.read_excel('input.xlsx', sheet_name=0, dtype={'SERIAL': str})
all_quota = pd.read_excel('input.xlsx', sheet_name=1)

# 处理样本条件
use_sample_cons = all_quota[all_quota['group'] == 99].iloc[0]
quota = all_quota[all_quota['group'] != 99].copy()

# 数据筛选
if use_sample_cons['condition'] == 't':
    filtered_data = data
else:
    try:
        py_cond = convert_condition(use_sample_cons['condition'])
        filtered_data = data.query(py_cond)
    except Exception as e:
        print(f"筛选条件错误: {e}")
        filtered_data = data

# 添加初始权重列
filtered_data['weight'] = 1.0  # 初始化权重为1

# 准备目标矩阵
quota['group_sep'] = quota.groupby('group').cumcount() + 1

trans = filtered_data[['SERIAL']].copy()
groups = quota['group'].unique()

for group in groups:
    trans[f'target{group}'] = np.nan
    group_quota = quota[quota['group'] == group]

    for _, row in group_quota.iterrows():
        try:
            cond = convert_condition(row['condition'])
            mask = filtered_data.eval(cond)
            trans.loc[mask, f'target{group}'] = row['group_sep']
        except Exception as e:
            print(f"条件处理错误: {row['condition']} - {str(e)}")

# 转换为分类类型并处理数据类型
target_columns = [f'target{g}' for g in groups]
for col in target_columns:
    trans[col] = pd.Categorical(trans[col])
    filtered_data[col] = trans[col].cat.codes  # 转换为分类代码

# 构建符合balance要求的targets结构 (关键修改点)
targets = {}
for group in groups:
    group_quota = quota[quota['group'] == group]
    # 创建分类到比例的映射
    category_map = (
        group_quota[['group_sep', 'target']]
        .set_index('group_sep')['target']
        .div(100)
        .to_dict()
    )
    # 转换为balance需要的结构
    targets[f'target{group}'] = {
        'categories': category_map,
        'type': 'categorical'
    }

# 创建样本对象
sample = Sample.from_frame(
    filtered_data,
    id_column="SERIAL",
    weight_column="weight",
    check_id_uniqueness=True
)

# 执行raking加权 (修正目标格式)
adjusted = sample.adjust(
    method="rake",
    target=targets,  # 使用新的目标结构
    max_iter=5000,
    weight_trimming_mean_ratio=320,
    convergence=0.0001,
    handle_unused_categories='skip'
)

# 计算最终权重
adjusted_df = adjusted.df
adjusted_df['wt'] = adjusted_df['weight'] * (use_sample_cons['target'] / len(data))

# 合并回原始数据
output_df = data.merge(
    adjusted_df[['SERIAL', 'wt']],
    on='SERIAL',
    how='left'
).fillna(0)

# 保存结果
output_df.to_csv('output.csv', index=False)

print("处理成功！关键指标：")
print(f"总权重: {output_df['wt'].sum():.2f} (目标: {use_sample_cons['target']})")
print(f"有效样本量: {(output_df['wt'].sum() ** 2 / (output_df['wt'] ** 2).sum()):.2f}")
