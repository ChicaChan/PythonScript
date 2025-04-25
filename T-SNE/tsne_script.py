import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import time
from joblib import Parallel, delayed
import multiprocessing

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_dir, exist_ok=True)

# 输入目录
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
os.makedirs(input_dir, exist_ok=True)


def load_data(file_path):
    """
    加载数据
    """
    print(f"正在加载数据: {file_path}...")
    try:
        df = pd.read_excel(file_path, index_col=0)
        # 检查是否为01数据
        if not all(df.isin([0, 1]).all()):
            print("警告: 数据中存在非01值，请检查数据格式")
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None


def calculate_kl_divergence(P, Q):
    """
    计算KL散度: KL(P||Q) = Σ P(i) * log(P(i)/Q(i))
    P: 高维空间中的概率分布
    Q: 低维空间中的概率分布
    """
    # 避免数值不稳定性
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    
    # 归一化
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    
    # 向量化计算KL散度 - 一次性计算所有元素
    kl_div = np.sum(P * np.log(P / Q))
    
    return kl_div


def compute_pairwise_probabilities(X, perplexity=30.0, metric='euclidean'):
    """
    计算高维空间中的条件概率分布
    """
    distances = pairwise_distances(X, metric=metric)
    n_samples = X.shape[0]
    
    # 设置对角线为0
    np.fill_diagonal(distances, 0)
    
    # 向量化计算条件概率
    # 使用广播计算所有点对的概率
    P = np.exp(-distances ** 2 / (2 * perplexity))
    
    # 设置对角线为0
    np.fill_diagonal(P, 0)
    
    # 归一化每一行
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / np.maximum(row_sums, 1e-10)  # 避免除以零
    
    # 对称化
    P = (P + P.T) / (2 * n_samples)
    
    return P


def calculate_q_distribution(X_tsne, metric='euclidean'):
    """
    计算低维空间的t分布概率
    """
    # 计算成对距离
    distances_low = pairwise_distances(X_tsne, metric=metric)
    
    # 设置对角线为0
    np.fill_diagonal(distances_low, 0)
    
    # 向量化计算t分布概率
    Q = 1 / (1 + distances_low ** 2)
    
    # 设置对角线为0
    np.fill_diagonal(Q, 0)
    
    # 归一化
    return Q / np.sum(Q)


def evaluate_tsne_params(X, perplexity_range, learning_rates, n_iter_range, metrics=['euclidean', 'manhattan'], n_jobs=-1, early_stopping_patience=10, random_state=42):
    """
    评估不同参数组合的效果
    参数:
        X: 输入数据
        perplexity_range: 困惑度参数范围
        learning_rates: 学习率范围
        n_iter_range: 迭代次数范围
        metrics: 距离度量方法
        n_jobs: 并行作业数，-1表示使用所有可用CPU
        early_stopping_patience: 早停耐心值，连续多少次迭代KL散度没有改善则停止
    """
    from joblib import Parallel, delayed
    import multiprocessing
    
    # 如果n_jobs为-1，使用所有可用的CPU核心数
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    results = []
    best_kl = float('inf')
    best_params = None
    best_tsne = None
    
    # 计算高维空间的概率分布
    P_high = compute_pairwise_probabilities(X)
    
    # 创建参数组合列表
    param_combinations = []
    for perplexity in perplexity_range:
        for learning_rate in learning_rates:
            for n_iter in n_iter_range:
                for metric in metrics:
                    param_combinations.append((perplexity, learning_rate, n_iter, metric))
    
    total_combinations = len(param_combinations)
    print(f"总共需要评估 {total_combinations} 种参数组合，使用 {n_jobs} 个CPU核心并行计算")
    
    # 定义单个参数组合的评估函数
    def evaluate_single_param_set(params, P_high, X):
        perplexity, learning_rate, n_iter, metric = params
        try:
            start_time = time.time()
            
            # t-SNE降维
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                metric=metric,
                random_state=42,
                n_iter_without_progress=early_stopping_patience  # 早停参数
            )
            X_tsne = tsne.fit_transform(X)
            
            # 向量化计算低维空间的概率分布
            Q = calculate_q_distribution(X_tsne, metric=metric)
            
            # 计算KL散度
            kl_divergence = calculate_kl_divergence(P_high, Q)
            
            elapsed_time = time.time() - start_time
            
            # 返回结果
            return {
                'perplexity': perplexity,
                'learning_rate': learning_rate,
                'n_iter': n_iter,
                'metric': metric,
                'kl_divergence': kl_divergence,
                'time': elapsed_time,
                'tsne': X_tsne,
                'success': True
            }
        except Exception as e:
            print(f"参数组合 {perplexity}, {learning_rate}, {n_iter}, {metric} 失败: {e}")
            return {
                'perplexity': perplexity,
                'learning_rate': learning_rate,
                'n_iter': n_iter,
                'metric': metric,
                'success': False,
                'error': str(e)
            }
    
    # 参数评估
    with tqdm(total=total_combinations, desc="参数调优进度") as pbar:
        # joblib并行计算
        parallel_results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_single_param_set)(params, P_high, X) for params in param_combinations
        )
        
        # 处理结果
        for result in parallel_results:
            pbar.update(1)
            if result.get('success', False):
                results.append(result)
                
                # 更新最佳参数
                if result['kl_divergence'] < best_kl:
                    best_kl = result['kl_divergence']
                    best_params = result.copy()
                    best_tsne = result['tsne']
    
    return results, best_params, best_tsne


def plot_tsne_results(X_tsne, labels=None, title="t-SNE降维结果", save_path=None):
    """
    可视化t-SNE降维结果
    """
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[colors[i]], label=f'类别 {label}', alpha=0.7)
        plt.legend()
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
    
    plt.title(title)
    plt.xlabel('t-SNE 维度 1')
    plt.ylabel('t-SNE 维度 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    plt.show()


def main():
    data_file = os.path.join(input_dir, 'data.xlsx')
    
    # 加载数据
    df = load_data(data_file)
    if df is None:
        return
    
    print(f"数据加载完成，shape: {df.shape}")
    
    # 数据预处理
    X = df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 定义超参数范围
    # 困惑度(默认为30,数据集越大,需要参数值越大,建议5-50)
    perplexity_range = [5,10,20]
    learning_rates = [100]
    n_iter_range = [1000]
    random_state = 42
    metrics = ['euclidean', 'manhattan']
    
    # 获取CPU核心数
    cpu_count = multiprocessing.cpu_count()
    n_jobs = max(1, cpu_count - 1)  # 保留一个核心
    
    print(f"系统检测到 {cpu_count} 个CPU核心，将使用 {n_jobs} 个核心进行并行计算")
    print("开始t-SNE参数调优...")
    
    # 添加早停参数，提高收敛速度
    early_stopping_patience = 10
    
    results, best_params, best_tsne = evaluate_tsne_params(
        X_scaled, 
        perplexity_range, 
        learning_rates, 
        n_iter_range, 
        metrics,
        random_state=random_state,
        n_jobs=n_jobs,
        early_stopping_patience=early_stopping_patience
    )
    
    # 输出最佳参数
    print("\n最佳参数组合:")
    print(f"困惑度 (perplexity): {best_params['perplexity']}")
    print(f"学习率 (learning_rate): {best_params['learning_rate']}")
    print(f"迭代次数 (n_iter): {best_params['n_iter']}")
    print(f"距离度量 (metric): {best_params['metric']}")
    print(f"KL散度: {best_params['kl_divergence']:.6f}")
    print(f"运行时间: {best_params['time']:.2f}秒")
    
    # 可视化最佳结果
    plot_tsne_results(
        best_tsne,
        title=f"t-SNE最佳结果 (KL散度: {best_params['kl_divergence']:.6f})",
        save_path=os.path.join(output_dir, 'tsne_best_result.png')
    )
    
    # 保存结果到Excel
    result_df = pd.DataFrame({
        'perplexity': [r['perplexity'] for r in results],
        'learning_rate': [r['learning_rate'] for r in results],
        'n_iter': [r['n_iter'] for r in results],
        'metric': [r['metric'] for r in results],
        'kl_divergence': [r['kl_divergence'] for r in results],
        'time': [r['time'] for r in results]
    })
    
    result_df = result_df.sort_values('kl_divergence')
    result_df.to_excel(os.path.join(output_dir, 'tsne_parameter_results.xlsx'), index=False)
    print(f"参数评估结果已保存至: {os.path.join(output_dir, 'tsne_parameter_results.xlsx')}")
    
    # 保存降维后的数据
    tsne_df = pd.DataFrame(
        best_tsne,
        index=df.index,
        columns=['tsne_1', 'tsne_2']
    )
    tsne_df.to_excel(os.path.join(output_dir, 'tsne_coordinates.xlsx'))
    print(f"降维坐标已保存至: {os.path.join(output_dir, 'tsne_coordinates.xlsx')}")


if __name__ == "__main__":
    main()