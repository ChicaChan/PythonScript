# -*- coding: utf-8 -*-
import pandas as pd
from relativeImp import relativeImp

df = pd.read_csv("D:\\workplace\\项目\\已完结\\LKK李锦记HK AIPL 24H1\\table\\table.csv", encoding='utf-8')
yName = 'B301'
xNames = ['C1T1W1',
          'C1T2W1',
          'C1T3W1',
          'C1T4W1',
          'C1T5W1',
          'C1T6W1',
          'C1T7W1',
          'C1T8W1',
          'C1T9W1',
          'C1T10W1',
          'C1T11W1',
          'C1T12W1',
          'C1T13W1',
          'C1T14W1',
          'C1T15W1',
          'C1T16W1',
          'C1T17W1',
          'C1T18W1',
          'C1T19W1',
          'C1T20W1',
          'C1T21W1',
          'C1T22W1',
          'C1T23W1',
          'C1T24W1',
          'C1T25W1',
          'C1T26W1',]

df_results = relativeImp(df, outcomeName=yName, driverNames=xNames)
print(df_results)

# 保存为Excel
df_results.to_excel("D:\\workplace\\项目\\已完结\\LKK李锦记HK AIPL 24H1\\table\\results.xlsx", index=False)
