# -*- coding: utf-8 -*-
import pandas as pd
from relativeImp import relativeImp

df = pd.read_csv("C:\\Users\\Qdtech\\Downloads\\字节支付NPS-25Q1-驱动数据-B12-C组.csv", encoding='utf-8')
yName = 'C001'
xNames = ['C1R1',
          'C2R1',
          'C4R1',
          'C6R1',
          'C9R1',
          'C11R1',
          'C13R1',
          'C16R1',
          'C18R1']

df_results = relativeImp(df, outcomeName=yName, driverNames=xNames).sort_values(by='rawRelaImpt', ascending=False)
print(df_results)
