import pandas as pd
import os

# 设置工作目录
os.chdir(r'd:\workplace\Python脚本\抽数')

# 读取数据表
print('数据表:')
df1 = pd.read_excel('input.xlsx', sheet_name=0)
print(df1.head())
print(f'数据表行数: {len(df1)}, 列数: {df1.shape[1]}')

# 读取条件表
print('\n条件表:')
df2 = pd.read_excel('input.xlsx', sheet_name=1)
print(df2)
print(f'条件表行数: {len(df2)}, 列数: {df2.shape[1]}')