import os
import pandas as pd
import re
from collections import defaultdict

# 初始化设置
pd.set_option('display.max_columns', None)
os.chdir(r"D:\DP小工具\2.转dat格式")  # 设置工作目录

filename = "data"  # 输入文件名（不含扩展名）

# ================= 读取CSV文件 =================
try:
    raw = pd.read_csv(
        f"./{filename}.csv",
        dtype=str,
        keep_default_na=False,
        encoding='gbk'
    )
except UnicodeDecodeError:
    raw = pd.read_csv(
        f"./{filename}.csv",
        dtype=str,
        keep_default_na=False,
        encoding='gb18030'
    )


# ================= 列名处理（._替换为0） =================
def sanitize_colname(col):
    """替换所有.和_为0，并处理连续特殊字符"""
    return re.sub(r'[._]+', '0', col)


raw.columns = [sanitize_colname(col) for col in raw.columns]


# ================= 类型检测逻辑 =================
def detect_column_type(series):
    """增强型类型检测"""
    cleaned = series.str.strip()

    # 严格整数检测（允许正负号）
    if cleaned.str.match(r'^[+-]?\d+$').all():
        return 'integer'

    # 严格数值检测（含小数点和科学计数法）
    try:
        temp = cleaned.str.replace(',', '', regex=False)
        if temp.str.match(r'^[+-]?\d+\.?\d*([eE][+-]?\d+)?$').all():
            pd.to_numeric(temp, errors='raise')
            return 'numeric'
    except:
        pass

    return 'character'


# 创建属性表
attribute = pd.DataFrame({
    'colname': raw.columns,
    'type': [detect_column_type(raw[col]) for col in raw.columns],
    'len': [6] * len(raw.columns)  # 初始化最小长度为6
})

# ================= 列宽计算（纯字符长度） =================
for i, col in enumerate(raw.columns):
    max_len = raw[col].apply(lambda x: len(str(x).strip())).max()
    attribute.at[i, 'len'] = max(6, max_len)  # 保证最小6位


# ================= 数据对齐处理 =================
def pad_column(series, width, col_type):
    """统一右对齐，空格填充"""
    series = series.astype(str).str.strip()

    # 所有类型右对齐
    return series.apply(
        lambda x: x.rjust(width) if len(x) <= width else x[:width]
    )


# 应用对齐处理
for col in raw.columns:
    col_info = attribute[attribute['colname'] == col].iloc[0]
    raw[col] = pad_column(raw[col], col_info['len'], col_info['type'])

# ================= 生成define.stp =================
define_lines = []
cum_pos = 1  # 起始位置

for _, row in attribute.iterrows():
    start = cum_pos
    end = cum_pos + row['len'] - 1
    prefix_map = {
        'integer': 'di',
        'numeric': 'dw',
        'character': 'dc'
    }
    define_lines.append(
        f"{prefix_map[row['type']]} ${row['colname']}=${start} - {end}"
    )
    cum_pos = end + 1  # 下一个字段起始位置

# ================= 生成make.stp =================
colname_counts = defaultdict(int)

base_names = [col.split('_')[0] for col in raw.columns]  # 已处理过列名
for name in base_names:
    colname_counts[name] += 1

    multi_choice = {k: v for k, v in colname_counts.items() if v > 1}
    make_lines = ["[*data ttl(;)=="]
    make_lines += [f"{name};{count};" for name, count in multi_choice.items()]
    make_lines += [
        "]",
        "[*do i=1:[ttl.#]/2]",
        "   [*do a=1:[ttl.i*2]]",
        "      om $[ttl.i*2-1]=$[ttl.i*2-1]0[a]/1-999,",
        "   [*end a]",
        "[*end i]"
    ]

# ================= 文件输出 =================
# 输出dat文件（固定宽度，右对齐）
with open(f"{filename}1.dat", 'w', encoding='gbk') as f:
    for _, row in raw.iterrows():
        f.write(''.join(row.values) + '\n')

# 输出配置文件
with open(f"{filename}define.stp", 'w', encoding='gbk') as f:
    f.write('\n'.join(define_lines))

with open(f"{filename}make.stp", 'w', encoding='gbk') as f:
    content = '\n'.join(make_lines) if multi_choice else "No multiple choice, please check your input data."
    f.write(content)

print("处理完成！字符型列信息：")
print(attribute[attribute['type'] == 'character'])