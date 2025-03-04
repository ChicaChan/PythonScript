import pandas as pd
import numpy as np

df = pd.read_excel('data2.xlsx')


# 处理多选题数据
def process_multichoice_data(data):
    if pd.isna(data):
        return 0
    elif data >= 1:
        return 1
    else:
        return data


# 初始化单选题和多选题列表
single_choice_columns = []
multi_choice_columns = []

for column in df.columns:
    if "_" in column:
        multi_choice_columns.append(column)
        df[column] = df[column].apply(process_multichoice_data)
    else:
        single_choice_columns.append(column)

# 输出匹配到的单选题和多选题
print("单选题:", single_choice_columns)
print("多选题:", multi_choice_columns)

df.to_excel('transformed_datav2.xlsx', index=False)
