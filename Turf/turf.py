# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import itertools
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib
import os
import tempfile

# 使用Agg后端以确保在无GUI环境下图片保存正确
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def turf_analysis(data, max_comb_size=3, top_combinations=3):
    """
    TURF分析核心算法
    参数:
        data: 包含0/1数据的DataFrame，每列代表一个产品，每行代表一个受访者
        max_comb_size: 最大组合产品数（组合中的产品数量上限）
        top_combinations: 每个组合大小保留的最佳组合数量

    返回:
        包含分析结果的字典，包含：
         - 产品总数
         - 受访者总数
         - 各组合大小的分析结果
         - 单产品覆盖率数据
    """
    # 数据有效性验证（必须为0/1数据）
    assert data.isin([0, 1]).all().all(), "输入文件必须为0/1数据"

    products = data.columns.tolist()  # 产品列表
    n_respondents = len(data)  # 样本总量
    results = {}  # 存储每个组合大小的结果

    single_coverage = data.sum()

    for k in range(1, max_comb_size + 1):
        combs = []
        # 生成所有可能的 k 产品组合
        total_combs = len(list(itertools.combinations(products, k)))
        iterator = itertools.combinations(products, k)
        if total_combs > 1000:
            iterator = tqdm(iterator, total=total_combs, desc=f'分析组合数 {k}')

        for comb in iterator:
            coverage = data[list(comb)].max(axis=1).sum()
            combs.append({
                'combination': comb,
                'coverage': coverage,
                'percentage': coverage / n_respondents  # 覆盖率百分比
            })

        sorted_combs = sorted(combs, key=lambda x: -x['coverage'])
        results[k] = {
            'top_combinations': sorted_combs[:top_combinations],
            'max_coverage': sorted_combs[0]['coverage'] if sorted_combs else 0
        }

    return {
        'n_products': len(products),
        'n_respondents': n_respondents,
        'results': results,
        'single_coverage': single_coverage
    }


def generate_turf_report(analysis_result, output_excel=None):
    """
    生成TURF分析报告和可视化图表
    参数:
        analysis_result: turf_analysis函数的返回结果
        output_excel: 输出Excel文件路径（None时直接显示图表）

    返回:
        格式化的文本报告字符串
    """
    cmb_sizes = sorted(analysis_result['results'].keys())  # 分析组合大小
    total = analysis_result['n_respondents']
    total_single = analysis_result['single_coverage'].sum()

    # 构建多列分布的报告文本列表，方便写入Excel时分栏显示
    report_lines = []
    separator = "=" * 70
    report_lines.append(separator)
    report_lines.append("              TURF 分析报告")
    report_lines.append(separator)
    report_lines.append(f"样本总量: {total}")
    report_lines.append(f"产品总数: {analysis_result['n_products']}")
    report_lines.append("")  # 空行

    report_lines.append("【单产品覆盖率】")
    single_cov = analysis_result['single_coverage'].sort_values(ascending=False)
    for product, cov in single_cov.items():
        report_lines.append(f"- {product}: {cov} ({cov / total:.1%})")
    report_lines.append("")  # 空行

    report_lines.append("【多产品组合分析】")
    # 用于图表绘制的数据
    plot_reach_rates = []  # 最大触达率
    plot_frequencies = []  # 最佳组合频次占比

    for k in cmb_sizes:
        res = analysis_result['results'][k]
        top_combos = res['top_combinations']
        if not top_combos:
            continue

        best_combo = top_combos[0]
        reach_rate = best_combo['coverage'] / total * 100  # 触达率百分比
        best_freq = sum(analysis_result['single_coverage'][prod] for prod in best_combo['combination'])
        best_frequency = (best_freq / total_single) * 100 if total_single > 0 else 0

        plot_reach_rates.append(reach_rate)
        plot_frequencies.append(best_frequency)

        report_lines.append(separator)
        report_lines.append(f"组合大小：{k}")
        report_lines.append(f"  -> 最佳组合到达率: {reach_rate:.1f}%   频次占比: {best_frequency:.1f}%")
        for i, comb in enumerate(top_combos):
            combo_freq = sum(analysis_result['single_coverage'][prod] for prod in comb['combination'])
            frequency = (combo_freq / total_single) * 100 if total_single > 0 else 0
            report_lines.append(
                f"      Top {i + 1}: 组合: {', '.join(comb['combination'])}  -> 到达率: {comb['percentage'] * 100:.1f}%, 频次: {frequency:.1f}%"
            )
    report_lines.append(separator)

    final_report = "\n".join(report_lines)
    print(final_report)

    # 绘制图表：触达率与频次占比随组合大小的变化
    plt.figure(figsize=(10, 6))
    plt.plot(cmb_sizes, plot_reach_rates, 'bo-', label='到达率 (%)', markersize=8)
    plt.plot(cmb_sizes, plot_frequencies, 'r^-', label='频次 (%)', markersize=8)
    plt.xlabel('组合中产品数量', fontsize=12)
    plt.ylabel('百分比 (%)', fontsize=12)
    plt.title('TURF 分析图表', fontsize=14)
    plt.xticks(cmb_sizes)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    for x, y in zip(cmb_sizes, plot_reach_rates):
        plt.text(x, y, f'{y:.1f}%', color='blue', fontsize=10, ha='center', va='bottom')
    for x, y in zip(cmb_sizes, plot_frequencies):
        plt.text(x, y, f'{y:.1f}%', color='red', fontsize=10, ha='center', va='top')

    if output_excel:
        # 保存图表到临时图片文件
        temp_image_path = os.path.join(tempfile.gettempdir(), "temp_chart.png")
        plt.savefig(temp_image_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 写入Excel文件，需要用xlsxwriter引擎
        writer = pd.ExcelWriter(output_excel, engine='xlsxwriter')
        workbook = writer.book

        # 定义格式
        format_title = workbook.add_format({'bold': True, 'font_color': 'blue', 'font_size': 14})
        format_cell = workbook.add_format({'text_wrap': True, 'valign': 'top', 'font_size': 11})

        # 创建报告工作表：将长报告按列分布，减轻单列压力
        report_ws = workbook.add_worksheet("报告")
        # 为了美观，将报告文本分成3列显示
        n_cols = 3
        n_lines = len(report_lines)
        col_height = (n_lines // n_cols) + (1 if n_lines % n_cols else 0)
        for idx, line in enumerate(report_lines):
            col = idx // col_height
            row = idx % col_height
            # 对于标题及分隔行使用高亮显示
            if (line.startswith("=") or "TURF 分析报告" in line):
                report_ws.write(row, col, line, format_title)
            else:
                report_ws.write(row, col, line, format_cell)
        # 自动调整列宽
        for i in range(n_cols):
            report_ws.set_column(i, i, 40)

        # 创建图表页，并插入图表图片
        chart_ws = workbook.add_worksheet("图表")
        chart_ws.insert_image('B2', temp_image_path, {'x_scale': 0.9, 'y_scale': 0.9})

        # 保存Excel文件，确保数据写入到文件中
        writer.close()
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
    else:
        plt.show()

    return final_report


if __name__ == "__main__":
    data = pd.read_csv("JK1_turf.csv", encoding='utf-8')
    valid_cols = data.columns[data.apply(lambda col: set(col.dropna().unique()).issubset({0, 1}))]
    data = data[valid_cols]

    analysis = turf_analysis(data, max_comb_size=5, top_combinations=3)

    generate_turf_report(analysis, output_excel="turf_report.xlsx")
