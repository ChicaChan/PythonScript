import pandas as pd

file_path = 'd:/workplace/Python脚本/抽数/input.xlsx'
print('Sheet names:', pd.ExcelFile(file_path).sheet_names)

data_df = pd.read_excel(file_path, sheet_name=0)
cond_df = pd.read_excel(file_path, sheet_name=1)

print('\nData sheet (first 5 rows):\n', data_df.head())
print('\nCondition sheet:\n', cond_df)