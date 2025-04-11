"""
外部数据处理脚本
功能：将多选题数据合并为单行数据,用分号隔开
"""
import pandas as pd
import re
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "input", "raw_data.xlsx")
output_file = os.path.join(script_dir, "output", "processed_data.xlsx")

df = pd.read_excel(input_file)
columns = df.columns

multi_choice_bases = set()
for col in columns:
    if re.search(r'_\d+$', col):
        base = re.sub(r'_\d+$', '', col)
        multi_choice_bases.add(base)

# 处理每个多选题
for base in multi_choice_bases:
    pattern = fr'^{re.escape(base)}_\d+$'
    related_cols = [col for col in columns if re.match(pattern, col)]
    
    def process_row(row):
        selected = []
        for col in related_cols:
            number = re.search(r'_(\d+)$', col)
            if number and pd.notna(row[col]):
                selected.append(number.group(1))
        return ';'.join(selected) if selected else ''

    if related_cols:
        df[base] = df[related_cols].apply(process_row, axis=1)
        df = df.drop(columns=related_cols)

df.to_excel(output_file, index=False)
print(f"处理完成，已保存到: {output_file}")