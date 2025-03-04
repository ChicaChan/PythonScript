import pandas as pd

file_path = 'data.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')
single_choice_columns = pd.read_excel(file_path, sheet_name='Sheet2', usecols=[0])
multiple_choice_columns = pd.read_excel(file_path, sheet_name='Sheet2', usecols=[1])


# 转化为01数据
def transform_multiple_choice(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype(str)
            dummies = pd.get_dummies(df[column].str.split(',', expand=True).stack()).groupby(level=0).sum()
            new_columns = [f"{column}{i + 1:02}" for i in range(dummies.shape[1])]
            dummies.columns = new_columns
            df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
    return df


transformed_data = transform_multiple_choice(data, multiple_choice_columns.iloc[:, 0])

single_choice_column_names = single_choice_columns.iloc[:, 0].tolist()
multiple_choice_column_names = multiple_choice_columns.iloc[:, 0].tolist()

print("单选题：", single_choice_column_names)
print("多选题：", multiple_choice_column_names)

output_file_path = 'transformed_data.xlsx'
transformed_data.to_excel(output_file_path, index=False)

print(f"数据已保存至 {output_file_path}")
