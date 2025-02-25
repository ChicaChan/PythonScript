import pandas as pd
import numpy as np
import re
from scipy.optimize import minimize
from numba import njit
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


def convert_condition(condition):
    """自动转换R风格条件到Python格式"""
    # 替换R语法元素（优化顺序）
    replacements = [
        (r"data\$(\w+)", r"\1"),  # 移除data$前缀，不添加反引号
        (r"\b==\b", " == "),  # 确保运算符空格
        (r"\b!=\b", " != "),
        (r"\b&\b", " and "),
        (r"\b\|\b", " or "),
        (r"%in%", ".isin"),
        (r"\bNA\b", "NaN")
    ]

    original_cond = condition.strip()

    # 执行替换
    for pattern, repl in replacements:
        condition = re.sub(pattern, repl, condition)

    # 转换字符串引号（单引号转双引号）
    condition = re.sub(r"'(\w+)'", r'"\1"', condition)

    # 添加安全括号
    if not condition.startswith("("):
        condition = f"({condition})"

    return condition, original_cond


def validate_data_structure(data_df, quota_df):
    """增强数据校验"""
    # 检查配额表结构
    if not {'group', 'condition', 'target'}.issubset(quota_df.columns):
        raise ValueError("配额表必须包含group/condition/target三列")


def build_joint_constraints(quota_df):
    """修正约束构建逻辑"""
    constraints = {}
    total_per_group = quota_df.groupby('group')['target'].sum()

    for _, row in quota_df.iterrows():
        group = int(row['group'])
        subgroup = int(row.name)  # 使用行索引作为subgroup

        # 计算组内比例（重要修正点）
        group_total = total_per_group[group]
        ratio = row['target'] / group_total

        constraints[(group, subgroup)] = ratio

    print("\n联合约束配置（组内比例）:")
    for (g, s), r in sorted(constraints.items()):
        print(f"Group {g}-{s}: {r:.4%}")
    return constraints


def create_indicator_matrix(trans_data, constraints):
    """生成指标矩阵时增加调试信息"""
    indicator_dict = {}
    print("\n指标矩阵维度检查:")

    for idx, ((group, subgroup)) in enumerate(constraints.keys(), 1):
        col_name = f'target_{group}'
        if col_name not in trans_data.columns:
            raise KeyError(f"转换矩阵缺少列: {col_name}")

        mask = trans_data[col_name] == subgroup
        print(f"约束 {idx}: Group {group}-{subgroup} => 匹配样本数: {mask.sum()}")

        indicator_dict[(group, subgroup)] = mask.astype(np.float64)

    matrix = np.column_stack(list(indicator_dict.values()))
    targets = np.array([constraints[k] for k in sorted(constraints.keys())])
    return matrix, targets


@njit
def fast_matrix_ops(X_T, weights):
    """数值稳定的矩阵运算"""
    weighted = X_T.dot(weights)
    total = np.sum(weights)
    return weighted / total if total > 1e-10 else np.zeros_like(weighted)


def calculate_weights_optimized(constraints, trans_data, max_attempts=15):
    """增强的优化过程"""
    X, y = create_indicator_matrix(trans_data, constraints)
    X_T = X.T.copy()
    n_samples = X.shape[0]

    # 打印关键参数
    print(f"\n优化参数详情:")
    print(f"样本数: {n_samples}")
    print(f"约束数: {X.shape[1]}")
    print(f"目标比例: {y}")

    def loss(params):
        actual = fast_matrix_ops(X_T, params)
        return np.sum((actual - y) ** 2) + 0.001 * np.var(params)

    best_weights = None
    best_loss = np.inf

    for attempt in range(max_attempts):
        current_cap = 1.5 + 0.5 * attempt
        bounds = [(0.1, current_cap)] * n_samples

        # 改进的初始化策略
        if attempt == 0:
            x0 = np.ones(n_samples)
        elif attempt % 3 == 0:
            x0 = np.random.uniform(0.5, 2.0, n_samples)
        else:
            x0 = best_weights * np.random.normal(1, 0.1, n_samples)
            x0 = np.clip(x0, 0.1, current_cap)

        # 使用更鲁棒的优化方法
        result = minimize(
            loss,
            x0=x0,
            method='trust-constr',
            bounds=bounds,
            options={'maxiter': 1000, 'xtol': 1e-8, 'gtol': 1e-8}
        )

        if result.success:
            current_loss = result.fun
            print(f"阶段{attempt + 1} [cap={current_cap:.1f}] 损失: {current_loss:.6f}")

            if current_loss < best_loss:
                best_loss = current_loss
                best_weights = result.x

            if current_loss < 1e-6:
                break
        else:
            print(f"阶段{attempt + 1} 未收敛: {result.message}")

    if best_weights is None:
        raise RuntimeError("所有优化尝试均失败，请检查约束条件是否冲突")

    # 标准化权重
    best_weights /= best_weights.mean()
    return best_weights


def main_process():
    try:
        # 数据读取
        print("读取输入文件中...")
        all_data = pd.read_excel("input.xlsx", sheet_name=0)
        all_quota = pd.read_excel("input.xlsx", sheet_name=1)

        # 数据校验
        validate_data_structure(all_data, all_quota)

        # 处理配额配置
        quota = all_quota[all_quota['group'] != 99].copy()
        use_sample_cons = all_quota[all_quota['group'] == 99].iloc[0]

        # 类型转换
        quota['group'] = quota['group'].astype(int)
        quota['target'] = pd.to_numeric(quota['target'], errors='coerce')
        target_value = float(use_sample_cons['target'])

        # 样本筛选
        if str(use_sample_cons['condition']).strip().lower() == 't':
            data = all_data.copy()
        else:
            condition = convert_condition(use_sample_cons['condition'])
            data = all_data.query(condition).copy()
        print(f"\n有效样本量: {len(data)}")

        # 创建转换矩阵
        trans = data[['SERIAL']].copy()
        groups = quota['group'].unique()

        for group in groups:
            trans[f'target_{group}'] = np.nan

        # 应用配额条件
        for idx, row in quota.iterrows():
            group = int(row['group'])
            condition = convert_condition(row['condition'])
            try:
                mask = data.eval(condition)
                trans.loc[mask, f'target_{group}'] = idx  # 使用行索引作为subgroup标识
            except Exception as e:
                raise ValueError(f"行{idx}条件解析失败: {condition}\n错误: {str(e)}")

        # 构建约束
        joint_constraints = build_joint_constraints(quota)

        # 计算权重
        print("\n开始权重优化...")
        weights = calculate_weights_optimized(joint_constraints, trans)

        # 应用权重
        total_sample = len(all_data)
        trans['weight'] = weights * (target_value / total_sample)

        # 合并结果
        output_data = all_data.merge(
            trans[['SERIAL', 'weight']],
            on='SERIAL',
            how='left'
        ).fillna({'weight': 0.0})

        # 结果验证
        print("\n最终权重统计:")
        print(f"总权重: {output_data['weight'].sum():.2f}")
        print(f"平均权重: {output_data['weight'].mean():.4f}")
        print(f"最大权重: {output_data['weight'].max():.4f}")
        print(f"最小权重: {output_data['weight'].min():.4f}")

        output_data.to_csv("optimized_output.csv", index=False)
        print("\n处理成功完成！")

    except Exception as e:
        print(f"\n错误发生: {str(e)}")
        print("故障排除建议:")
        print("1. 检查YUEHUO08和PF列是否存在且类型正确")
        print("2. 确认所有条件表达式中的拼写正确")
        print("3. 验证group=99的target值为有效数字")
        print("4. 检查是否所有约束条件都有匹配样本")


if __name__ == "__main__":
    main_process()
