# encoding=utf-8
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import warnings
import matplotlib

matplotlib.use('TkAgg')

warnings.filterwarnings("ignore")

df = pd.read_excel('D:\\办公软件\\DP小工具\\myproject\\聚类数据源1101.xlsx')
# print(data.head())

x = df.iloc[:, 2:].values
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
# print(x_scaled)

# # 计算wcss
# wcss = []
# for k in range(1, 31):
#     kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=100, random_state=40)
#     kmeans.fit(x_scaled)
#     wcss.append(kmeans.inertia_)
#
# # print(wcss)
#
# plt.figure(figsize=(10, 5))
# plt.grid()
# plt.plot(range(1, 31), wcss, marker='o')
# plt.show()


# 计算轮廓系数
silhouette_scores = []
for k in range(5, 30):
    kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=100, random_state=42)
    kmeans.fit(x_scaled)
    score = silhouette_score(x_scaled, kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(10, 5))
plt.grid()
plt.plot(range(5, 30), silhouette_scores, marker='o')
plt.show()



