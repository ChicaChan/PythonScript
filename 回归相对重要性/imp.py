# -*- coding: utf-8 -*-
import pandas as pd
from relativeImp import relativeImp

df = pd.read_csv("回归相对重要性\\table.csv", encoding='utf-8')
# 筛选数据
#df = df[df['SERIAL'] <= 100]


yName = 'B301'

# 使用其余的变量作为解释变量
xNames = list(df.columns)
xNames.remove(yName)
xNames.remove('SERIAL')          

df_results = relativeImp(df, outcomeName=yName, driverNames=xNames)

# 只保存normalizedImp
df_results = df_results[['driver','normRelaImpt']]
print(df_results)

# 保存为Excel
df_results.to_excel("回归相对重要性\\results.xlsx", index=False)
