import pandas as pd

# 读取Excel文件
file_path = "D:\\workplace\\项目\\花西子\\花西子table\\tableW8-v1- spss\\25Q1花西子中期-多期.xlsx"  
df = pd.read_excel(file_path, dtype=str) 

# 替换所有的"#NULL!"为""
df = df.replace("#NULL!", "")

df.to_excel(file_path, index=False)