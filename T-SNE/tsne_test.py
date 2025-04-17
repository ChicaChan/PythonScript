import numpy as np
import time
import matplotlib
import argparse
import sys
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

# 设置中文字体，防止中文乱码和警告
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 命令行参数解析
parser = argparse.ArgumentParser(description='PCA与t-SNE降维对比可视化')
parser.add_argument('--n_class', type=int, default=10, help='分类数')
parser.add_argument('--tsne_metric', type=str, default='euclidean', choices=['euclidean', 'manhattan'], help='t-SNE距离度量')
parser.add_argument('--tsne_perplexity', type=float, default=30, help='t-SNE困惑度')
parser.add_argument('--tsne_random_state', type=int, default=0, help='t-SNE随机种子')
parser.add_argument('--dataset', type=str, default='digits', choices=['digits', 'iris'], help='数据集选择')
args = parser.parse_args()

N_CLASS = args.n_class
TSNE_METRIC = args.tsne_metric
TSNE_PERPLEXITY = args.tsne_perplexity
TSNE_RANDOM_STATE = args.tsne_random_state
DATASET = args.dataset

# 加载数据集
try:
    print(f"加载数据集: {DATASET} ...")
    if DATASET == 'digits':
        digits = datasets.load_digits(n_class=N_CLASS)
        X = digits.data
        y = digits.target
    elif DATASET == 'iris':
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        N_CLASS = len(np.unique(y))
    else:
        raise ValueError('暂不支持的数据集')
    n_samples, n_features = X.shape
except Exception as e:
    print(f"数据集加载失败: {e}")
    sys.exit(1)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA降维对比
print("PCA降维...")
t0 = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_time = time.time() - t0

# t-SNE降维
print(f"t-SNE降维(metric={TSNE_METRIC})...")
t0 = time.time()
try:
    tsne = TSNE(n_components=2, init='pca', random_state=TSNE_RANDOM_STATE, metric=TSNE_METRIC, perplexity=TSNE_PERPLEXITY)
    X_tsne = tsne.fit_transform(X_scaled)
    tsne_time = time.time() - t0
except Exception as e:
    print(f"t-SNE降维失败: {e}")
    sys.exit(1)

# 可视化
plt.figure(figsize=(12, 5))

# t-SNE可视化
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=15, alpha=0.7)
plt.title(f't-SNE (metric={TSNE_METRIC})\n耗时: {tsne_time:.2f}s')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
cbar1 = plt.colorbar(scatter1, ticks=range(N_CLASS))
cbar1.set_label('类别')
plt.grid(True, linestyle='--', alpha=0.3)

# PCA可视化
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=15, alpha=0.7)
plt.title(f'PCA\n耗时: {pca_time:.2f}s')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
cbar2 = plt.colorbar(scatter2, ticks=range(N_CLASS))
cbar2.set_label('类别')
plt.grid(True, linestyle='--', alpha=0.3)

plt.suptitle('t-SNE vs PCA 降维可视化对比', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 输出运行时间
print(f"PCA降维耗时: {pca_time:.2f}秒")
print(f"t-SNE降维耗时: {tsne_time:.2f}秒")