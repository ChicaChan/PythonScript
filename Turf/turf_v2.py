# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import os
import tempfile

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def turf_analysis(data, max_comb_size=10):
    """
    改进的TURF分析算法，采用逐步贪心策略
    """
    assert data.isin([0, 1]).all().all(), "输入数据必须为0/1格式"

    products = data.columns.tolist()
    n = len(data)
    selected = []
    results = []
    cumulative_coverage = pd.Series(0, index=data.index)

    for k in range(1, max_comb_size + 1):
        best_add = None
        best_coverage = 0
        best_combination = []
        remaining = [p for p in products if p not in selected]

        for p in tqdm(remaining, desc=f'分析组合大小 {k}'):
            temp_comb = selected + [p]
            coverage = data[temp_comb].max(axis=1).sum()
            if coverage > best_coverage:
                best_coverage = coverage
                best_add = p
                best_combination = temp_comb

        if best_add:
            selected.append(best_add)
            cumulative_coverage = cumulative_coverage | data[best_add]

            freq = data[selected].sum().sum()
            total_freq = data.sum().sum()

            results.append({
                '组大小': k,
                '新增变量': best_add,
                '保留变量': selected[:-1],
                '到达率': best_coverage,
                '个案百分比': f"{best_coverage / n:.0%}",
                '频率': freq,
                '响应百分比': f"{freq / total_freq:.0%}"
            })

    return {
        '总样本量': n,
        '总频率': data.sum().sum(),
        '结果': pd.DataFrame(results),
        '最终组合': selected
    }


def generate_turf_report(analysis_result, output_file="turf_report.xlsx"):
    """
    生成符合SPSS模板格式的报告
    """
    df = analysis_result['结果']
    n = analysis_result['总样本量']
    total_freq = analysis_result['总频率']

    # 创建格式化表格
    report_df = pd.DataFrame({
        '变量': ['ADDED: ' + row['新增变量'] + '\nKEPT: ' + ', '.join(row['保留变量'])
                 if row['组大小'] > 1 else 'ADDED: ' + row['新增变量']
                 for _, row in df.iterrows()],
        '统计': [f"{row['到达率']}\n{row['个案百分比']}" for _, row in df.iterrows()],
        '组大小': df['组大小'],
        '到达率': df['到达率'],
        '个案百分比': [f"{int(row['到达率'] / n * 100)}%" for _, row in df.iterrows()],
        '频率': df['频率'],
        '响应百分比': [f"{int(row['频率'] / total_freq * 100)}%" for _, row in df.iterrows()]
    })

    # 创建Excel报告
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    workbook = writer.book

    # 数据表格式
    format_header = workbook.add_format({
        'bold': True,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'bg_color': '#D9E1F2'
    })

    format_data = workbook.add_format({
        'text_wrap': True,
        'valign': 'top',
        'border': 1
    })

    # 写入数据
    report_df.to_excel(writer, sheet_name='TURF分析', index=False, startrow=2)
    worksheet = writer.sheets['TURF分析']

    # 设置列宽
    worksheet.set_column('A:A', 35)
    worksheet.set_column('B:B', 15)
    worksheet.set_column('C:G', 12)

    # 添加标题
    title_format = workbook.add_format({
        'bold': True,
        'font_size': 16,
        'align': 'center',
        'valign': 'vcenter'
    })
    worksheet.merge_range('A1:G1', '最佳到达率及频率（按组大小排列）', title_format)

    # 添加表头格式
    for col_num, value in enumerate(report_df.columns.values):
        worksheet.write(2, col_num, value, format_header)

    # 添加数据格式
    for row_num in range(3, len(report_df) + 3):
        for col_num in range(7):
            worksheet.write(row_num, col_num, report_df.iloc[row_num - 3, col_num], format_data)

    # 创建图表
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    df['到达率'].plot(kind='line', marker='o', ax=ax, color='#4472C4', linewidth=2)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['组大小'])
    ax.set_xlabel('组大小')
    ax.set_ylabel('到达人数')
    ax.set_title('到达率趋势图')
    ax.grid(True, linestyle='--', alpha=0.6)

    # 保存图表
    chart_path = os.path.join(tempfile.gettempdir(), 'turf_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 插入图表
    worksheet.insert_image('I3', chart_path)

    writer.close()
    os.remove(chart_path)

    print(f"报告已生成: {output_file}")
    return report_df


if __name__ == "__main__":
    data = pd.read_csv("JK1_turf.csv", encoding='utf-8')
    valid_cols = data.columns[data.apply(lambda col: set(col.dropna().unique()).issubset({0, 1}))]
    data = data[valid_cols]
    analysis = turf_analysis(data, max_comb_size=29)
    report = generate_turf_report(analysis)
