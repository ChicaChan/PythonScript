import os
import pandas as pd
import re
from collections import defaultdict

# 初始化设置
pd.set_option('display.max_columns', None)
os.chdir(r"D:\DP小工具\2.转dat格式")

filename = "data"

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


# ================= 列名处理 =================
def sanitize_colname(col):
    return re.sub(r'[._]+', '0', col)


raw.columns = [sanitize_colname(col) for col in raw.columns]


# ================= 判断字段类型 =================
def detect_column_type(series):
    cleaned = series.str.strip()

    # 空值处理
    if cleaned.empty:
        return 'character'

    # 整数检测
    if cleaned.str.match(r'^[+-]?\d+$').all():
        return 'integer'

    # 数值检测（含科学计数法）
    try:
        temp = cleaned.str.replace(',', '', regex=False)
        pd.to_numeric(temp, errors='raise')
        # 排除纯整数
        if temp.str.contains(r'\.|e|E', regex=True).any():
            return 'numeric'
        return 'integer'
    except:
        pass

    return 'character'


# 属性表
attribute = pd.DataFrame({
    'colname': raw.columns,
    'type': [detect_column_type(raw[col]) for col in raw.columns],
    'len': [6] * len(raw.columns)  # 初始化最小长度
})

# ================= 列宽计算 =================
for i, col in enumerate(raw.columns):
    max_len = raw[col].apply(lambda x: len(str(x).strip())).max()
    attribute.at[i, 'len'] = max(6, max_len)


# ================= 数据对齐 =================
def pad_column(series, width):
    return series.str.strip().apply(
        lambda x: x.rjust(width)
    )


for col in raw.columns:
    col_len = attribute[attribute['colname'] == col]['len'].values[0]
    raw[col] = pad_column(raw[col].astype(str), col_len)

# ================= 生成define.stp =================
define_lines = []
cum_pos = 1  # 当前字段起始位置

for _, row in attribute.iterrows():
    start = cum_pos
    end = cum_pos + row['len'] - 1
    # 类型映射
    prefix = {
        'integer': 'di',
        'numeric': 'dw',
        'character': 'dc'
    }[row['type']]
    define_lines.append(
        f"{prefix} ${row['colname']}=${start}-{end},"
    )
    cum_pos = end + 2  # 字段宽度 + 1个空格分隔符

# ================= 生成make.stp =================
# 多选题识别逻辑
pattern = re.compile(r'^(.*?)(0\d+)+$')
base_counts = defaultdict(int)

for col in raw.columns:
    match = pattern.match(col)
    if match:
        base_name = match.group(1)
        base_counts[base_name] += 1

# 过滤有效多选字段（出现次数>1）
multi_choice = {k: v for k, v in base_counts.items() if v > 1}

# 构建make.stp内容
make_lines = ["[*data ttl(;)=="]
for name, count in multi_choice.items():
    make_lines.append(f"{name};{count};")
make_lines += [
    "]",
    "[*do i=1:[ttl.#]/2]",
    "   [*do a=1:[ttl.i*2]]",
    "      om $[ttl.i*2-1]=$[ttl.i*2-1]0[a]/1-999,",
    "   [*end a]",
    "[*end i]"
]

# ================= 文件输出 =================
# 输出dat文件
with open(f"{filename}1.dat", 'w', encoding='gbk') as f:
    for _, row in raw.iterrows():
        line = ' '.join(row.values)  # 字段间添加空格
        f.write(line + '\n')

# 输出配置文件
with open(f"{filename}1define.stp", 'w', encoding='gbk') as f:
    f.write('\n'.join(define_lines))

with open(f"{filename}1make.stp", 'w', encoding='gbk') as f:
    if multi_choice:
        f.write('\n'.join(make_lines))
    else:
        f.write("No multiple choice fields detected")

print("处理完成！字符型列信息：")
print(attribute['type'].value_counts())
print(attribute[attribute['type'] == 'character'])
