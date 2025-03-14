# -*- coding: utf-8 -*-
import pandas as pd
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
    TURF分析，采用逐步贪心策略
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
    生成报告
    """
    df = analysis_result['结果']
    n = analysis_result['总样本量']
    total_freq = analysis_result['总频率']

    # 创建格式化表格
    report_df = pd.DataFrame({
        '变量': ['ADDED: ' + row['新增变量'] + '\nKEPT: ' + ', '.join(row['保留变量'])
                 if row['组大小'] > 1 else 'ADDED: ' + row['新增变量']
                 for _, row in df.iterrows()],
        # '统计': [f"{row['到达率']}\n{row['个案百分比']}" for _, row in df.iterrows()],
        '组大小': df['组大小'],
        '到达率': df['到达率'],
        '个案百分比': [f"{row['到达率'] / n * 100:.1f}%" for _, row in df.iterrows()],
        '频率': df['频率'],
        '响应百分比': [f"{row['频率'] / total_freq * 100:.1f}%" for _, row in df.iterrows()]
    })

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
        'valign': 'vcenter',
        'align': 'center',
        'border': 1
    })

    report_df.to_excel(writer, sheet_name='TURF分析', index=False, startrow=2)
    worksheet = writer.sheets['TURF分析']

    # 列宽
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

    # 表头格式
    for col_num, value in enumerate(report_df.columns.values):
        worksheet.write(2, col_num, value, format_header)

    # 数据格式
    for row_num in range(3, len(report_df) + 3):
        for col_num in range(6):
            worksheet.write(row_num, col_num, report_df.iloc[row_num - 3, col_num], format_data)

    plt.figure(figsize=(14, 6.5))
    ax = plt.subplot(111)

    # 计算百分比
    reach_percent = (df['到达率'] / n) * 100
    response_percent = (df['频率'] / total_freq) * 100

    # 合并坐标轴
    all_values = pd.concat([reach_percent, response_percent])
    y_min, y_max = 0, all_values.max() * 1.1

    # 绘制双线
    reach_line = ax.plot(reach_percent, marker='o', color='#4472C4',
                         linewidth=2, label='到达率')
    response_line = ax.plot(response_percent, marker='s', color='#ED7D31',
                            linewidth=2, label='响应百分比')

    # 坐标轴格式
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['组大小'])
    plt.subplots_adjust(bottom=0.15)
    ax.set_xlabel('组大小', rotation=30)
    ax.set_ylabel('百分比')
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper left')
    plt.title('G图', pad=20)
    ax.grid(True, linestyle='--', alpha=0.6)

    chart_path = os.path.join(tempfile.gettempdir(), 'turf_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    chart_worksheet = workbook.add_worksheet('G图')
    chart_worksheet.insert_image('A1', chart_path)

    # 数据表列宽
    chart_worksheet.set_column('A:A', 50)

    writer.close()
    os.remove(chart_path)

    print(f"报告已生成: {output_file}")
    return report_df


if __name__ == "__main__":
    data = pd.read_csv("data(多变量).csv", encoding='utf-8')
    # 统计data的01字段个数
    col_count = data.apply(lambda col: set(col.dropna().unique()).issubset({0, 1})).sum()
    valid_cols = data.columns[data.apply(lambda col: set(col.dropna().unique()).issubset({0, 1}))]
    data = data[valid_cols]
    analysis = turf_analysis(data, max_comb_size=col_count)
    report = generate_turf_report(analysis)
