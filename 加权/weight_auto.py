import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import sys
from pathlib import Path


def safe_convert_condition(cond, columns):
    cond = str(cond).strip().replace("data$", "").replace("`", "")
    cond = re.sub(r"=='(\w+)'", r"== '\1'", cond)
    cond = re.sub(r'=="(\w+)"', r'== "\1"', cond)
    cond = re.sub(r'%in%c\(([^)]+)\)', r'.isin([\1])', cond)
    cond = re.sub(r'(?<![=!<>])=(?!=)', ' == ', cond)
    cond = re.sub(r'(?<!\s)(&)(?!\s)', ' & ', cond)
    cond = re.sub(r'(?<!\s)(\|)(?!\s)', ' | ', cond)

    tokens = []
    for token in re.split(r'(\W)', cond):
        token = token.strip()
        if not token: continue
        if token in columns and not re.match(r"^['\"]", token):
            tokens.append(f"`{token}`" if re.search(r"\s", token) else token)
        else:
            tokens.append(token)

    cond = "".join(tokens)
    open_paren = cond.count("(")
    close_paren = cond.count(")")
    if open_paren != close_paren:
        cond += ")" * (open_paren - close_paren) if open_paren > close_paren else "(" * (close_paren - open_paren)
    return f"({cond})" if not cond.startswith("(") else cond


class FastRakingOptimizer:
    def __init__(self, df, conditions, total_sample):
        self.df = df.copy()
        self.conditions = []
        for raw_cond, target in conditions:
            converted = safe_convert_condition(raw_cond, df.columns)
            try:
                test_mask = df.eval(converted, engine='python')
                self.conditions.append((converted, target))
            except Exception as e:
                raise ValueError(f"条件转换失败: {raw_cond} -> {converted}\n错误: {str(e)}")
        self.total = total_sample
        self.n = len(df)
        self.masks = self._precompute_masks()

    def _precompute_masks(self):
        masks = []
        for cond, _ in self.conditions:
            mask = self.df.eval(cond, engine='python').values
            if mask.sum() == 0:
                raise ValueError(f"'{cond}' 无匹配样本")
            masks.append(mask)
        return masks

    def optimize(self, max_iter=500, tol=1e-6, cap=3.0, growth_factor=1.1):
        weights = np.full(self.n, self.total / self.n, dtype=np.float64)
        targets = np.array([c[1] for c in self.conditions], dtype=np.float64)
        best_weights = weights.copy()
        min_error = np.inf

        for epoch in tqdm(range(max_iter), desc="优化权重"):
            max_adj = 0
            current_cap = cap * (growth_factor ** (epoch // 10))

            for i, mask in enumerate(self.masks):
                current = np.dot(weights, mask)
                if current < 1e-10: continue
                ratio = (targets[i] / current) ** 0.8
                adjustment = np.where(mask, ratio, 1.0)
                weights *= adjustment
                weights = np.clip(weights, 0, current_cap)
                max_adj = max(max_adj, abs(ratio - 1))

            weights *= self.total / weights.sum()
            errors = [abs(np.dot(weights, m) - t) / t for m, t in zip(self.masks, targets)]
            current_error = np.max(errors)

            if current_error < min_error:
                min_error = current_error
                best_weights = weights.copy()
            if current_error < tol or max_adj < 1e-5: break

        return best_weights


def process_single_pair(data_df, cond_df, data_sheet_name, cond_sheet_name):
    """处理单个数据"""
    try:
        total_sample = cond_df[cond_df['group'] == 99]['target'].iloc[0]
    except IndexError:
        raise ValueError(f"配额表 {cond_sheet_name} 缺少目标样本总量")

    conditions = []
    target_proportions = []
    for _, row in cond_df[cond_df['group'] != 99].iterrows():
        target_pct = row['target'] / 100  # 转换为比例
        target_abs = target_pct * total_sample
        conditions.append((row['condition'], target_abs))
        target_proportions.append(target_pct)

    optimizer = FastRakingOptimizer(data_df, conditions, total_sample)
    data_df['weight'] = optimizer.optimize(max_iter=500, tol=1e-6, cap=1.5, growth_factor=1.05)

    # 生成验证报告
    validation = []
    total_abs_diff = 0
    total_pct_diff = 0
    for i, ((cond, target_abs), target_pct) in enumerate(zip(conditions, target_proportions)):
        actual_abs = np.dot(data_df['weight'], optimizer.masks[i])
        actual_pct = actual_abs / total_sample
        abs_diff = actual_pct - target_pct
        pct_diff = (abs_diff / target_pct * 100) if target_pct != 0 else 0

        validation.append({
            "条件": cond[:50],
            "目标比例": f"{target_pct:.8%}",
            "实际比例": f"{actual_pct:.8%}",
            "绝对差异": f"{abs_diff:+.8%}",
            "相对差异": f"{pct_diff:+.8f}%"
        })
        total_abs_diff += abs(abs_diff)
        total_pct_diff += abs(pct_diff)

    return data_df, {
        "数据表": data_sheet_name,
        "配额表": cond_sheet_name,
        "总样本量": total_sample,
        "总绝对差异": f"{total_abs_diff:.8%}",
        "平均相对差异": f"{total_pct_diff / len(conditions):.8f}%",
        "详细验证": validation
    }


def process_multi_sheets(input_file):
    """处理多Sheet文件"""
    all_sheets = pd.read_excel(input_file, sheet_name=None)
    sheet_names = list(all_sheets.keys())
    results = {}
    reports = []

    for i in range(0, len(sheet_names), 2):
        if i + 1 >= len(sheet_names):
            print(f"警告: 跳过未配对的数据表 {sheet_names[i]}")
            continue

        data_sheet = sheet_names[i]
        cond_sheet = sheet_names[i + 1]

        try:
            df, report = process_single_pair(
                all_sheets[data_sheet].copy(),
                all_sheets[cond_sheet].copy(),
                data_sheet,
                cond_sheet
            )
            results[f"{data_sheet}_加权"] = df
            reports.append(report)
        except Exception as e:
            print(f"处理失败 {data_sheet}|{cond_sheet}: {str(e)}")
            continue

    return results, reports


def print_report(reports):
    """验证报告"""
    for report in reports:
        print(f"\n=== {report['数据表']} 加权结果 ===")
        print(f"配额表: {report['配额表']}")
        print(f"总样本量: {report['总样本量']}")
        print(f"总绝对差异: {report['总绝对差异']}")
        # print(f"平均相对差异: {report['平均相对差异']}")

        print("\n详细验证:")
        print(f"{'条件':<50} | {'目标':<8} | {'实际':<8} | {'绝对差异':<10}")
        # print(f"{'条件':<50} | {'目标':<8} | {'实际':<8} | {'绝对差异':<10} | {'相对差异':<8}")
        print("-" * 90)
        for v in report['详细验证']:
            print(
                f"{v['条件']:<50} | {v['目标比例']:<8} | {v['实际比例']:<8} | {v['绝对差异']:<10}")
            """
            print(
                f"{v['条件']:<50} | {v['目标比例']:<8} | {v['实际比例']:<8} | {v['绝对差异']:<10} | {v['相对差异']:<8}")
"""


# 修改主程序部分
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("参数错误，使用方法：")
        print("weight.py [输入路径] [输出路径]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"输入文件不存在：{input_path}")
        sys.exit(1)

    try:
        # 处理文件
        results, reports = process_multi_sheets(str(input_path))
        print_report(reports)

        # 保存结果
        with pd.ExcelWriter(output_path) as writer:
            for sheet_name, df in results.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print(f"\n结果已保存至：{output_path}")

    except Exception as e:
        print(f"\n处理过程中发生错误：{str(e)}")
        sys.exit(1)

