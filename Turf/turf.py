# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import itertools
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']


def turf_analysis(data, max_comb_size=3, top_combinations=3):
    """
    TURF分析

    参数：
    data: DataFrame 受访者-产品矩阵（0/1）
    max_comb_size: int 最大组合尺寸
    top_combinations: int 每个尺寸保留的最佳组合数量

    返回：
    分析结果字典
    """
    # 数据校验
    assert data.isin([0, 1]).all().all(), "输入数据必须为0/1矩阵"

    products = data.columns.tolist()
    n_respondents = len(data)

    results = {}

    # 预计算所有单项覆盖
    single_coverage = data.sum()

    # 主分析循环
    for k in range(1, max_comb_size + 1):
        combs = []

        # 生成所有可能的组合
        candidates = itertools.combinations(products, k)
        total_combs = len(list(itertools.combinations(products, k)))

        # 使用进度条（当组合数>1000时）
        iterator = itertools.combinations(products, k)
        if total_combs > 1000:
            iterator = tqdm(iterator, total=total_combs, desc=f'分析组合尺寸 {k}')

        # 评估每个组合
        for comb in iterator:
            coverage = data[list(comb)].max(axis=1).sum()
            combs.append({
                'combination': comb,
                'coverage': coverage,
                'percentage': coverage / n_respondents
            })

        # 按覆盖度排序
        sorted_combs = sorted(combs, key=lambda x: -x['coverage'])

        # 保存前N个组合
        results[k] = {
            'top_combinations': sorted_combs[:top_combinations],
            'max_coverage': sorted_combs[0]['coverage']
        }

    return {
        'n_products': len(products),
        'n_respondents': n_respondents,
        'results': results,
        'single_coverage': single_coverage
    }


def generate_turf_report(analysis_result, output_file=None):
    """
    生成TURF分析报告和可视化图表

    参数：
    analysis_result: turf_analysis 返回的分析结果
    output_file: 输出文件路径（可选）

    返回：
    格式化文本报告和包含两个曲线的图表：
    - 到达率曲线：展示不同组合尺寸下的最大覆盖率
    - 频率曲线：展示组合中产品被选择的总频率
    """
    import matplotlib.pyplot as plt

    cmb_sizes = sorted(analysis_result['results'].keys())
    total = analysis_result['n_respondents']
    total_single = analysis_result['single_coverage'].sum()

    reach_rates = []
    report = []
    report.append("=" * 60)
    report.append("TURF 分析报告")
    report.append("=" * 60)
    report.append(f"受访者数量: {total}")
    report.append(f"产品数量: {analysis_result['n_products']}\n")
    report.append("单项覆盖分析：")
    single_cov = analysis_result['single_coverage'].sort_values(ascending=False)
    for product, cov in single_cov.items():
        report.append(f"- {product}: {cov} ({cov / total:.1%})")
    report.append("\n组合覆盖分析：")

    plot_reach_rates = []
    plot_frequencies = []

    for k in cmb_sizes:
        res = analysis_result['results'][k]
        top_combos = res['top_combinations']
        if not top_combos:
            continue

        best_combo = top_combos[0]
        reach_rate = best_combo['coverage'] / total * 100
        best_freq = sum(analysis_result['single_coverage'][prod] for prod in best_combo['combination'])
        best_frequency = (best_freq / total_single) * 100

        plot_reach_rates.append(reach_rate)
        plot_frequencies.append(best_frequency)

        report.append(f"\n组合尺寸 {k}: 到达率 {reach_rate:.1f}% ，频率 {best_frequency:.1f}%")
        for i, comb in enumerate(top_combos):
            combo_freq = sum(analysis_result['single_coverage'][prod] for prod in comb['combination'])
            frequency = (combo_freq / total_single) * 100
            report.append(
                f"  Top {i + 1}: 组合: {', '.join(comb['combination'])} - 到达率: {comb['percentage'] * 100:.1f}%，频率: {frequency:.1f}%"
            )

    plt.figure(figsize=(8, 5))
    plt.plot(cmb_sizes, plot_reach_rates, 'bo-', label='到达率')
    plt.plot(cmb_sizes, plot_frequencies, 'r^-', label='频率')
    plt.xlabel('组合数')
    plt.ylabel('百分比 (%)')
    plt.title('到达率与频率分析图')
    plt.legend()
    plt.grid(True)

    final_report = "\n".join(report)
    print(final_report)

    if output_file:
        plt.savefig(output_file.replace('.txt', '.png'))
    plt.close()
    return final_report


if __name__ == "__main__":
    data = pd.read_csv("data.csv", encoding='utf-8')

    valid_cols = data.columns[data.apply(lambda col: set(col.dropna().unique()).issubset({0, 1}))]
    data = data[valid_cols]
    analysis = turf_analysis(data, max_comb_size=12, top_combinations=5)
    generate_turf_report(analysis, output_file="turf_report.txt")
