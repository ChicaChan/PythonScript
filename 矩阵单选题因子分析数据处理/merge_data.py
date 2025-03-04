import pandas as pd

df = pd.read_excel('data.xlsx')


def process_column_name(col_name):
    # 检查列名是否包含 '_'
    if '_' in col_name:
        # 拆分列名，获取题号.品牌序号_描述
        question_brand, description = col_name.split('_')
        question = question_brand.split('.')[0]
        return f"{question}_{description}"
    else:
        return col_name


merged_columns = {}
no_underscore_columns = []


for col in df.columns:
    new_col = process_column_name(col)

    if new_col != col:
        if new_col not in merged_columns:
            merged_columns[new_col] = df[col]
        else:
            merged_columns[new_col] += df[col]
    else:
        no_underscore_columns.append(col)

merged_df = pd.DataFrame(merged_columns)

for col in no_underscore_columns:
    merged_df[col] = df[col]

if no_underscore_columns:
    print("没有'_'的列有：")
    print(no_underscore_columns)

print("合并后的数据：")
print(merged_df.head())

# 检查哪些列的总和为0
zero_sum_columns = merged_df.columns[merged_df.sum() == 0].tolist()

# 输出总和为0的列
if zero_sum_columns:
    print("总和为0的列有：")
    print(zero_sum_columns)
else:
    print("没有总和为0的列。")

# 保存结果到新文件
merged_df.to_csv('merged_data.csv', index=False)

print("数据处理完成，结果已保存为 'merged_data.csv'")
