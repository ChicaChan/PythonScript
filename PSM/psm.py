import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import matplotlib

# 更换后端为 TkAgg
matplotlib.use('TkAgg')

# 读取文件
excel_file = pd.ExcelFile('data.xlsx')

# 获取指定工作表中的数据
df = excel_file.parse('Sheet1')

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 200

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 设置画布大小
plt.figure(figsize=(6, 3))

# 提取绘图数据
x = df['price']
y1 = df['Too Espensive']
y2 = df['Espensive']
y3 = df['Cheap']
y4 = df['Too Cheap']

# 绘制四条直线
plt.plot(x, y1, label='太贵')
plt.plot(x, y2, label='有点贵')
plt.plot(x, y3, label='有点便宜')
plt.plot(x, y4, label='太便宜')

# 设置图表标题和坐标轴标签
plt.title('价格与累计百分比关系图')
plt.xlabel('价格')
plt.xticks(rotation=45)
plt.ylabel('累计百分比')

# 添加图例
plt.legend()

# 显示图表
plt.show()


# 定义函数计算交点
def find_intersection(x1, y1, x2, y2):
    f1 = interp1d(x1, y1, kind='cubic')
    f2 = interp1d(x2, y2, kind='cubic')

    def difference(x):
        return f1(x) - f2(x)

    try:
        result = root_scalar(difference, bracket=[min(x1.min(), x2.min()), max(x1.max(), x2.max())])
        if result.converged:
            return result.root, f1(result.root)
    except ValueError:
        pass

    return None, None


# 计算两两之间的交点
lines = [('太贵', y1), ('有点贵', y2), ('有点便宜', y3), ('太便宜', y4)]
intersections = {}
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
        name1, line1 = lines[i]
        name2, line2 = lines[j]
        x_intersect, y_intersect = find_intersection(x, line1, x, line2)
        if x_intersect is not None and y_intersect is not None:
            intersections[(name1, name2)] = (x_intersect, y_intersect)

print('两两直线之间的交点：')
for (name1, name2), (x_intersect, y_intersect) in intersections.items():
    print(f'{name1} 和 {name2} 的交点：价格 = {x_intersect}, 累计百分比 = {y_intersect}')