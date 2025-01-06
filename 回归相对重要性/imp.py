# -*- coding: utf-8 -*-
import pandas as pd
from relativeImp import relativeImp

df = pd.read_excel("C:\\Users\\Qdtech\\Downloads\\table-v2.xlsx")
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

df_results = relativeImp(df, outcomeName=yName, driverNames=xNames)
print(df_results)
