import pandas as pd
import numpy as np
from tqdm import tqdm
import re


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
        if not token:
            continue
        if token in columns and not re.match(r"^['\"]", token):
            tokens.append(f"`{token}`" if re.search(r"\s", token) else token)
        else:
            tokens.append(token)

    cond = "".join(tokens)
    open_paren = cond.count("(")
    close_paren = cond.count(")")
    if open_paren != close_paren:
        diff = open_paren - close_paren
        cond += ")" * diff if diff > 0 else "(" * (-diff)
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
        for cond, target in self.conditions:
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
                if current < 1e-10:
                    continue
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
            if current_error < tol or max_adj < 1e-5:
                break

        return best_weights


def process_single_data(df, cond_df, data_sheet_name, cond_sheet_name):
    """加权单个数据"""
    try:
        total_sample = cond_df[cond_df['group'] == 99]['target'].iloc[0]
    except IndexError:
        raise ValueError(f"条件表 {cond_sheet_name} 缺少group=99的总量定义")

    conditions = []
    target_proportions = []
    for _, row in cond_df[cond_df['group'] != 99].iterrows():
        target_proportion = row['target'] / 100
        target_abs = target_proportion * total_sample
        conditions.append((row['condition'], target_abs))
        target_proportions.append(target_proportion)

    optimizer = FastRakingOptimizer(df, conditions, total_sample)
    df['weight'] = optimizer.optimize(max_iter=500, tol=1e-6, cap=1.5, growth_factor=1.05)

    validation = []
    total_diff = 0.0
    total_target_proportion = 0.0
    for i, ((cond, target_abs), (raw_cond, _)) in enumerate(zip(optimizer.conditions, conditions)):
        actual_abs = np.dot(df['weight'], optimizer.masks[i])
        actual_proportion = actual_abs / total_sample
        target_proportion = target_proportions[i]

        diff_proportion = actual_proportion - target_proportion
        relative_diff = (diff_proportion / target_proportion * 100) if target_proportion != 0 else 0.0

        total_diff += abs(diff_proportion)
        total_target_proportion += target_proportion

        validation.append({
            "条件表": cond_sheet_name,
            "条件表达式": raw_cond,
            "目标比例": target_proportion,
            "实际比例": actual_proportion,
            "绝对差异": diff_proportion,
            "相对差异 (%)": relative_diff
        })

    total_relative_diff = (total_diff / total_target_proportion * 100) if total_target_proportion != 0 else 0.0

    return df, {
        "数据表": data_sheet_name,
        "条件表": cond_sheet_name,
        "总样本量": total_sample,
        "总绝对差异（比例）": total_diff,
        "总相对差异 (%)": total_relative_diff,
        "详细验证": validation
    }


def process_multi_sheets(input_file):
    """处理多Sheet文件并返回结果和验证数据"""
    all_sheets = pd.read_excel(input_file, sheet_name=None)
    sheet_names = list(all_sheets.keys())
    results = {}
    validation_report = []

    for i in range(0, len(sheet_names), 2):
        if i + 1 >= len(sheet_names):
            print(f"[警告] 跳过未配对的数据表 {sheet_names[i]}")
            continue

        data_sheet = sheet_names[i]
        cond_sheet = sheet_names[i + 1]
        print(f"\n[处理中] {data_sheet} + {cond_sheet}")

        try:
            weighted_df, validation = process_single_data(
                all_sheets[data_sheet].copy(),
                all_sheets[cond_sheet].copy(),
                data_sheet,
                cond_sheet
            )
            results[f"{data_sheet}_加权"] = weighted_df
            validation_report.append(validation)
        except Exception as e:
            print(f"[错误] 处理失败 {data_sheet}|{cond_sheet}: {str(e)}")
            continue

    return results, validation_report


def print_validation_report(validation_report):
    """验证报告"""
    for report in validation_report:
        print(f"\n=== 验证结果 [{report['数据表']}] ===")
        print(f"总样本量: {report['总样本量']}")
        print(f"总绝对差异（比例）: {report['总绝对差异（比例）']:.8f}")
        print(f"总相对差异: {report['总相对差异 (%)']:.8f}%")

        print("\n详细条件验证:")
        for detail in report['详细验证']:
            print(f"{detail['条件表达式'][:50]:<50} | "
                  f"目标: {detail['目标比例']:>7.8%} | "
                  f"实际: {detail['实际比例']:>7.8%} | "
                  f"差异: {detail['绝对差异']:+7.8%} "
                  f"({detail['相对差异 (%)']:+.8f}%)")


if __name__ == "__main__":
    results, validation = process_multi_sheets("input-24Q3信任感rep-new.xlsx")

    print("\n" + "=" * 80)
    print_validation_report(validation)

    with pd.ExcelWriter("output.xlsx") as writer:
        for sheet_name, df in results.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    print("\n[完成] 结果已保存至 output.xlsx")
