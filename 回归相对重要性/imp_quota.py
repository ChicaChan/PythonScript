# -*- coding: utf-8 -*-
import pandas as pd
from relativeImp import relativeImp

# 读取数据
df = pd.read_csv("回归相对重要性\\data.csv", encoding='gbk')

# 创建ExcelWriter对象，准备写入多个Sheet
with pd.ExcelWriter("回归相对重要性\\quota_results.xlsx", engine="openpyxl") as writer:
    # 遍历所有的W和BRAND_C1
    for BRAND_B1 in range(1, 14):
        df_sub = df[(df['W'] == 2) & (df['BRAND_B1'] == BRAND_B1)]
        yName = 'P12M'
        # 筛选除了P12M的所有变量
        xNames = list(df_sub.columns)
        xNames.remove(yName)
        xNames.remove('W')
        xNames.remove('BRAND_B1')
        df_results = relativeImp(df_sub, outcomeName=yName, driverNames=xNames)
        print(df_results)
        # 写入对应Sheet
        df_results.to_excel(writer, sheet_name=f"QUOTA_{BRAND_B1}", index=False)

