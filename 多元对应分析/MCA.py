import pandas as pd
import prince
import matplotlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



data = pd.read_excel("data.xlsx")

df = pd.DataFrame(data)

# One-Hot编码，排除ID列
df_encoded = pd.get_dummies(df.iloc[:, 1:])

# 多元对应分析
mca = prince.MCA(n_components=2, random_state=42)
mca = mca.fit(df_encoded)

# 获取MCA的主成分坐标
coordinates = mca.transform(df_encoded)

plt.figure(figsize=(8, 6))
plt.scatter(coordinates[0], coordinates[1])

# 标注每个点
for i, id_val in enumerate(df['userID']):
    plt.text(coordinates[0][i], coordinates[1][i], str(id_val), fontsize=12)

plt.title('Multiple Correspondence Analysis (MCA) - Plot')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.show()
