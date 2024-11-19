import pandas as pd
import openpyxl

# 读取Excel文件
a = pd.read_excel("data.xlsx", sheet_name=0, header=None)
a.replace("-", 0, inplace=True)

# 找到第一个值为 "Abs" 的单元格的行索引
abs_row_index = a[a.apply(lambda x: x.str.contains('Abs', na=False)).any(axis=1)].index[0]

# 品牌所在行
brand_row = abs_row_index - 1

# 提取品牌
# 提取第一列的所有非空数据
brand = a.iloc[brand_row, 1:].values
first_column_data = a.iloc[:, 0].dropna().values

# 提取属性（除去前两个非空单元格）
# 去除掉前两个数据
attr = a.iloc[2:, 0].dropna().values
attr = first_column_data[2:]

# 确保品牌和属性的长度一致
min_length = min(len(brand), len(attr))
brand = brand[:min_length]
attr = attr[:min_length]

# 打印DataFrame形状和min_length
print(f"DataFrame形状: {a.shape}")
print(f"最小长度: {min_length}")

# 创建初始数据框
df1 = pd.DataFrame({'brand': brand, 'attr': attr})

# 循环添加其他品牌的数据
for i in range(1, a.shape[1] + 1):
    if i < a.shape[1]:  # 检查i是否在范围内
        num = a.iloc[2:, i].dropna().values[:min_length]  # 确保num的长度与brand和attr一致
        df = pd.DataFrame({'brand': brand, 'attr': attr, 'num': num})
        df1 = pd.concat([df1, df], ignore_index=True)
    else:
        print(f"列索引 {i} 超出范围")

# 写入Excel文件
df1.to_excel("output.xlsx", index=False)
