import pandas as pd
import numpy as np
from scipy.optimize import minimize


def convert_condition(condition_str):
    """改进的条件转换函数"""
    return (
        condition_str.replace("data$", "")
        .replace("&", " and ").replace("|", " or ")
        .replace("==", " == ").replace("!=", " != ")
        .strip()
    )


def detect_primary_key(data_df):
    """安全的主键检测"""
    for col in data_df.columns:
        if data_df[col].nunique() == len(data_df):
            return col
    return data_df.columns[0]


def build_dynamic_constraints(quota_df):
    """约束条件生成（仅使用quota表的group）"""
    constraints = {}
    for group in quota_df['group'].unique():
        group_data = quota_df[quota_df['group'] == group]
        total = group_data['target'].sum()

        for _, row in group_data.iterrows():
            key = f"G{group}_S{row['group_sep']}"
            constraints[key] = row['target'] / total
    return constraints


def calculate_weights(constraints, trans_data):
    """优化后的权重计算"""
    categories = sorted(constraints.keys())
    target_ratios = np.array([constraints[cat] for cat in categories])

    # 矩阵加速优化
    indicator_matrix = pd.get_dummies(trans_data['target_group'])[categories].values.T

    def loss(params):
        weighted = indicator_matrix.dot(params)
        total = params.sum()
        return np.sum((weighted / total - target_ratios) ** 2)

    # 优化参数设置
    weights = np.ones(len(trans_data))
    result = minimize(
        loss,
        x0=weights,
        method='L-BFGS-B',
        bounds=[(0.1, None)] * len(weights),
        options={'maxiter': 500, 'ftol': 1e-6}
    )
    return result.x


# 主程序流程
try:
    # 数据读取
    print("正在读取数据...")
    all_data = pd.read_excel("input.xlsx", sheet_name=0)
    all_quota = pd.read_excel("input.xlsx", sheet_name=1)

    # 列校验（关键修正点）
    required_quota_cols = {'group', 'target', 'condition'}
    missing_quota_cols = required_quota_cols - set(all_quota.columns)
    if missing_quota_cols:
        raise KeyError(f"配额表缺少必要列: {missing_quota_cols}")

    # 动态生成分组序号（使用quota表的group）
    quota = all_quota[all_quota['group'] != 99].copy()
    quota['group_sep'] = quota.groupby('group').cumcount() + 1

    # 主键处理（仅数据表）
    primary_key = detect_primary_key(all_data)
    print(f"主键列确认: {primary_key}")

    # 样本筛选
    use_sample_cons = all_quota[all_quota['group'] == 99].iloc[0]
    if use_sample_cons['condition'].lower().strip() == 't':
        data = all_data.copy()
    else:
        condition = convert_condition(use_sample_cons['condition'])
        data = all_data.query(condition).copy()
        print(f"应用筛选条件: {condition} | 剩余样本: {len(data)}")

    # 创建转换矩阵（仅使用数据表字段）
    trans = pd.DataFrame({
        primary_key: data[primary_key].astype(str),
        'target_group': 'missing'
    })

    # 应用配额条件（从quota表获取group信息）
    for _, row in quota.iterrows():
        condition = convert_condition(row['condition'])
        try:
            mask = data.eval(condition)
            trans.loc[mask, 'target_group'] = f"G{row['group']}_S{row['group_sep']}"
        except Exception as e:
            raise ValueError(f"条件解析失败: {condition} | 错误: {str(e)}")

    # 处理未覆盖样本
    missing_mask = trans['target_group'] == 'missing'
    print(f"\n未覆盖样本数: {missing_mask.sum()}")
    trans = trans[~missing_mask].reset_index(drop=True)

    # 构建约束
    cons_dict = build_dynamic_constraints(quota)
    print("\n动态约束比例:")
    for k, v in cons_dict.items():
        print(f"{k}: {v:.4f}")

    # 计算权重
    print("\n开始优化计算...")
    weights = calculate_weights(cons_dict, trans)

    # 应用权重
    trans['weight'] = weights * (use_sample_cons['target'] / len(all_data))
    output_data = data.merge(trans[[primary_key, 'weight']], on=primary_key)
    output_data.to_csv("optimized_output.csv", index=False)

    # 验证结果
    print("\n验证结果:")
    weighted_total = output_data['weight'].sum()
    for group in quota['group'].unique():
        group_data = quota[quota['group'] == group]
        total_target = group_data['target'].sum()

        for _, row in group_data.iterrows():
            key = f"G{group}_S{row['group_sep']}"
            mask = trans['target_group'] == key
            actual = trans.loc[mask, 'weight'].sum() / weighted_total
            target = row['target'] / total_target
            print(f"{key}: 目标={target:.4f} 实际={actual:.4f}")

    # 有效样本量
    n_eff = (weighted_total ** 2) / (output_data['weight'] ** 2).sum()
    print(f"\n有效样本量: {n_eff:.1f}")

except Exception as e:
    print(f"\n运行时错误: {str(e)}")
    print("排查建议：")
    print("1. 确认配额表包含 group/target/condition 三列")
    print("2. 检查条件表达式中的字段是否存在于数据表")
    print("3. 验证group=99的行是否设置总样本量")
