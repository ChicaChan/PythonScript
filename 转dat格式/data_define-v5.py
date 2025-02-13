import os
import pandas as pd
import numpy as np
import re
from collections import defaultdict

# 初始化设置
pd.set_option('display.max_columns', None)
os.chdir(r"D:\DP小工具\2.转dat格式")

filename = "data"
raw = pd.read_csv(
    f"./{filename}.csv",
    dtype=str,
    keep_default_na=False,
    encoding='gbk'
)


# ================= 列名处理 =================
def sanitize_colname(col):
    """替换所有特殊字符为0"""
    return re.sub(r'[._]', '0', col)


raw.columns = [sanitize_colname(col) for col in raw.columns]


# ================= 类型检测 =================
def detect_column_type(series):
    cleaned = series.str.strip()

    # 检测整数（允许前导/尾随空格）
    if cleaned.str.match(r'^[+-]?\d+$').all():
        return 'integer'

    # 检测数值（含科学计数法和千分位）
    try:
        temp = cleaned.str.replace(',', '', regex=False)
        pd.to_numeric(temp, errors='raise')
        return 'numeric'
    except:
        pass

    return 'character'


attribute = pd.DataFrame({
    'colname': raw.columns,
    'type': [detect_column_type(raw[col]) for col in raw.columns],
    'len': [6] * len(raw.columns)
})


# ================= 计算显示宽度 =================
def get_display_width(s):
    """考虑中文字符的显示宽度（中文占2个字符位）"""
    s = str(s)
    chinese_count = len(re.findall(r'[\u4e00-\u9fff]', s))
    return len(s) + chinese_count


for i, col in enumerate(raw.columns):
    max_width = raw[col].apply(get_display_width).max()
    attribute.at[i, 'len'] = max(6, max_width)


# ================= 数据对齐 =================
def pad_column(series, width, col_type):
    """根据列类型进行对齐"""
    series = series.astype(str)

    if col_type in ['integer', 'numeric']:
        # 数值类型：右对齐，前补空格
        return series.str.strip().apply(lambda x: x.rjust(width)
        )
    else:
        # 字符类型：左对齐，后补空格
        return series.apply(lambda x: x.rjust(width))


for col in raw.columns:
    col_info = attribute[attribute['colname'] == col].iloc[0]
    raw[col] = pad_column(raw[col], col_info['len'], col_info['type'])

# ================= 生成define.stp =================
define_lines = []
cum_pos = 1

for _, row in attribute.iterrows():
    start = cum_pos
    end = cum_pos + row['len'] - 1
    prefix = {
        'integer': 'di',
        'numeric': 'dw',
        'character': 'dc'
    }[row['type']]

    define_lines.append(
        f"{prefix} ${row['colname']}=${start}-{end},"
    )
    cum_pos = end + 1  # 字段间无间隔

# ================= 生成make.stp =================
colname_counts = defaultdict(int)
base_names = [col.split('_')[0] for col in raw.columns]
for name in base_names:
    colname_counts[name] += 1

multi_choice = {k: v for k, v in colname_counts.items() if v > 1}
make_lines = ["[*data ttl(;)=="]
for name, count in multi_choice.items():
    make_lines.append(f"{name};{count};")
make_lines += ["]", "[*do i=1:[ttl.#]/2]",
               "   [*do a=1:[ttl.i*2]]",
               "      om $[ttl.i*2-1]=$[ttl.i*2-1]0[a]/1-999,",
               "   [*end a]", "[*end i]"]

# ================= 输出文件 =================
# 生成dat文件（固定宽度格式）
with open(f"{filename}1.dat", 'w', encoding='gbk') as f:
    for _, row in raw.iterrows():
        line = ''.join(row.values)
        f.write(line + '\n')

# 生成配置文件
with open(f"{filename}define.stp", 'w', encoding='gbk') as f:
    f.write('\n'.join(define_lines))

with open(f"{filename}make.stp", 'w', encoding='gbk') as f:
    content = '\n'.join(make_lines) if multi_choice else "No multiple choice,please check your input data."
    f.write(content)

print("处理完成！字符型列信息：")
print(attribute[attribute['type'] == 'character'])