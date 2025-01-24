import pandas as pd
import numpy as np
from ipfn import ipfn  # 用于实现rim weighting
import warnings
warnings.filterwarnings('ignore')

# 读取数据
def rim_weight():
    # 读取输入文件
    all_data = pd.read_excel("input.xlsx", sheet_name=0)
    all_quota = pd.read_excel("input.xlsx", sheet_name=1)
    
    # 分离目标样本量和配额条件
    use_sample_cons = all_quota[all_quota['group'] == '99']
    quota = all_quota[all_quota['group'] != '99']
    
    # 筛选加权样本
    data = all_data
    if use_sample_cons['condition'].iloc[0] != 't':
        # 使用eval安全地执行条件筛选
        condition = use_sample_cons['condition'].iloc[0]
        data = data.query(condition)
    
    # 按照组别重新生成加权数据
    unique_groups = quota['group'].unique()
    ng = len(unique_groups)
    
    # 创建转换矩阵
    trans = pd.DataFrame()
    trans[data.columns[0]] = data[data.columns[0]]
    
    # 为每个组创建target列
    for i in range(ng):
        trans[f'target{i+1}'] = None
    
    # 组内排序
    group_counts = quota['group'].value_counts().sort_index()
    group_sep = []
    for count in group_counts:
        group_sep.extend(range(1, count + 1))
    quota['group_sep'] = group_sep
    
    # 填充条件数据
    for _, row in quota.iterrows():
        mask = data.eval(row['condition'])
        trans.loc[mask, f'target{row["group"]}'] = row['group_sep']
    
    # 转换为因子(分类变量)
    for col in trans.columns[1:]:
        trans[col] = trans[col].astype('category')
        if trans[col].isna().any():
            print(f"第{col[6:]}组存在不满足条件的数据")
    
    # 准备目标比例
    targets_dict = {}
    for group in unique_groups:
        group_data = quota[quota['group'] == group]
        targets = group_data['target'].values / 100
        targets_dict[f'target{group}'] = dict(zip(range(1, len(targets) + 1), targets))
    
    # 执行加权
    dimensions = []
    target_margins = []
    
    for i in range(ng):
        col = f'target{i+1}'
        dimensions.append(trans[col].values)
        target_margins.append(list(targets_dict[col].values()))
    
    # 初始化权重
    weights = np.ones(len(data))
    
    # 使用ipfn进行迭代计算
    IPF = ipfn.ipfn(dimensions, target_margins, weights, max_iteration=1000)
    weights = IPF.iteration()
    
    # 调整权重使其符合目标样本量
    target_sample = use_sample_cons['target'].iloc[0]
    weights = weights * (target_sample / len(all_data))
    
    # 输出结果
    output_sample = data.copy()
    output_sample['wt'] = weights
    
    # 保存结果
    output_sample.to_csv("output.csv", index=False)
    
    # 计算加权效果指标
    weighting_effect = np.sum(weights)**2 / np.sum(weights**2)
    print(f"加权效果指标(越接近未加权样本量越好): {weighting_effect}")

if __name__ == "__main__":
    rim_weight()