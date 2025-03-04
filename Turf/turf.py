# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import itertools
from matplotlib import pyplot as plt
from tqdm import tqdm


def turf_analysis(data, max_comb_size=5, top_combinations=3):
    """
    TURF分析核心算法

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
    生成TURF分析报告

    参数：
    analysis_result: turf_analysis的输出结果
    output_file: 输出文件路径（可选）
    """
    # 基础信息
    report = []
    report.append("=" * 60)
    report.append("TURF分析报告")
    report.append("=" * 60)
    report.append(f"受访者数量: {analysis_result['n_respondents']}")
    report.append(f"产品数量: {analysis_result['n_products']}\n")

    # 单项覆盖分析
    report.append("单项覆盖分析：")
    single_cov = analysis_result['single_coverage'].sort_values(ascending=False)
    for product, cov in single_cov.items():
        report.append(f"- {product}: {cov} ({cov / analysis_result['n_respondents']:.1%})")

    # 组合分析
    report.append("\n组合覆盖分析：")
    for k in analysis_result['results']:
        res = analysis_result['results'][k]
        report.append(f"\n组合尺寸 {k}:")
        report.append(
            f"最大覆盖人数: {res['max_coverage']} ({res['max_coverage'] / analysis_result['n_respondents']:.1%})")

        for i, comb in enumerate(res['top_combinations']):
            report.append(f"Top {i + 1}:")
            report.append(f"  组合: {', '.join(comb['combination'])}")
            report.append(f"  覆盖人数: {comb['coverage']} ({comb['percentage']:.1%})")

    # 可视化
    plt.figure(figsize=(10, 6))
    x = list(analysis_result['results'].keys())
    y = [v['max_coverage'] for v in analysis_result['results'].values()]
    plt.plot(x, y, 'bo-')
    plt.title('Max Coverage by Combination Size')
    plt.xlabel('Combination Size')
    plt.ylabel('Coverage')
    plt.grid(True)

    # 输出报告
    final_report = '\n'.join(report)
    print(final_report)
    plt.show()

    # 保存文件
    if output_file:
        with open(output_file, 'w') as f:
            f.write(final_report)
        plt.savefig(output_file.replace('.txt', '.png'))

    return final_report


# 示例用法
if __name__ == "__main__":
    # 生成模拟数据（100个受访者，10个产品）
    np.random.seed(42)
    data = pd.DataFrame(np.random.choice([0, 1], size=(100, 10), p=[0.7, 0.3]),
                        columns=[f'Product_{chr(65 + i)}' for i in range(10)])

    # 运行TURF分析
    analysis = turf_analysis(data, max_comb_size=5)

    # 生成报告
    generate_turf_report(analysis)
