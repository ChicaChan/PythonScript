import pandas as pd
import numpy as np
from scipy.optimize import minimize


def convert_condition(condition_str):
    return (
        condition_str.replace("data$", "")
        .replace("&", " and ").replace("|", " or ")
        .replace("==", " == ").replace("!=", " != ")
        .strip()
    )


def detect_primary_key(data_df):
    for col in data_df.columns:
        if data_df[col].nunique() == len(data_df):
            return col
    return data_df.columns[0]


def build_dynamic_constraints(quota_df):
    constraints = {}
    for group in quota_df['group'].unique():
        group_data = quota_df[quota_df['group'] == group]
        total_target = group_data['target'].sum()
        group_constraint = {}
        for _, row in group_data.iterrows():
            subgroup = row['group_sep']
            ratio = row['target'] / total_target
            group_constraint[subgroup] = ratio
        col_name = f'target_{group}'
        constraints[col_name] = group_constraint
    return constraints


def calculate_weights(constraints, trans_data, initial_cap=1.5):
    cap = initial_cap
    max_iterations = 10
    constraint_cols = list(constraints.keys())

    # Prepare indicator matrices
    indicator_matrices = {}
    target_ratios = {}
    for col in constraint_cols:
        subgroups = list(constraints[col].keys())
        valid_mask = trans_data[col].notna()
        dummies = pd.get_dummies(trans_data.loc[valid_mask, col].astype(int))[subgroups]
        indicator_matrices[col] = (valid_mask, dummies)
        target_ratios[col] = np.array([constraints[col][s] for s in subgroups])

    def loss(params):
        total_loss = 0.0
        for col in constraint_cols:
            valid_mask, dummies = indicator_matrices[col]
            valid_weights = params[valid_mask]
            subgroup_weights = dummies.T.dot(valid_weights)
            subgroup_total = valid_weights.sum()
            if subgroup_total == 0:
                return np.inf
            actual_ratios = subgroup_weights / subgroup_total
            target = target_ratios[col]
            total_loss += np.sum((actual_ratios - target) ** 2)
        return total_loss

    # Cap adjustment loop
    for _ in range(max_iterations):
        bounds = [(0.1, cap)] * len(trans_data)
        result = minimize(
            loss,
            x0=np.ones(len(trans_data)),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-6}
        )
        if result.success:
            return result.x
        cap += 1
    raise RuntimeError("无法在最大迭代次数内找到解")


# 主程序流程
try:
    # 数据读取
    all_data = pd.read_excel("input.xlsx", sheet_name=0)
    all_quota = pd.read_excel("input.xlsx", sheet_name=1)

    # 动态生成分组序号
    quota = all_quota[all_quota['group'] != 99].copy()
    quota['group_sep'] = quota.groupby('group').cumcount() + 1

    # 主键处理
    primary_key = detect_primary_key(all_data)

    # 样本筛选
    use_sample_cons = all_quota[all_quota['group'] == 99].iloc[0]
    if use_sample_cons['condition'].lower().strip() == 't':
        data = all_data.copy()
    else:
        condition = convert_condition(use_sample_cons['condition'])
        data = all_data.query(condition).copy()

    # 创建转换矩阵
    groups = quota['group'].unique()
    trans = pd.DataFrame({
        primary_key: data[primary_key].astype(str)
    })
    for group in groups:
        trans[f'target_{group}'] = np.nan

    # 应用配额条件
    for _, row in quota.iterrows():
        group = row['group']
        col_name = f'target_{group}'
        condition = convert_condition(row['condition'])
        mask = data.eval(condition)
        trans.loc[mask, col_name] = row['group_sep']

    # 检查缺失值
    for group in groups:
        na_count = trans[f'target_{group}'].isna().sum()
        if na_count > 0:
            print(f"警告: group {group} 中有 {na_count} 个样本未覆盖条件")

    # 构建约束
    cons_dict = build_dynamic_constraints(quota)

    # 计算权重
    weights = calculate_weights(cons_dict, trans, initial_cap=1.5)

    # 应用权重调整
    total_samples = len(all_data)
    trans['weight'] = weights * (use_sample_cons['target'] / total_samples)

    # 输出结果
    output_data = data.merge(trans[[primary_key, 'weight']], on=primary_key)
    output_data.to_csv("optimized_output.csv", index=False)

    # 验证结果
    weighted_total = output_data['weight'].sum()
    for group in groups:
        group_data = quota[quota['group'] == group]
        total_target = group_data['target'].sum()

        for _, row in group_data.iterrows():
            subgroup = row['group_sep']
            mask = trans[f'target_{group}'] == subgroup
            actual = trans.loc[mask, 'weight'].sum() / weighted_total
            target = row['target'] / total_target
            print(f"Group {group}-{subgroup}: 目标={target:.4f} 实际={actual:.4f}")

    # 有效样本量
    n_eff = (weighted_total ** 2) / (output_data['weight'] ** 2).sum()
    print(f"有效样本量: {n_eff:.1f}")

except Exception as e:
    print(f"运行时错误: {str(e)}")
    print("排查建议:")
    print("1. 确认配额表包含 group/target/condition 三列")
    print("2. 检查条件表达式中的字段是否存在于数据表")
    print("3. 验证group=99的行是否设置总样本量")
