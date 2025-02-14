import os
import pandas as pd
import re
from collections import defaultdict

# ================= 初始化设置 =================
pd.set_option('display.max_columns', None)
os.chdir(r"D:\DP小工具\2.转dat格式")
filename = "data"

# ================= 读取原始数据 =================
try:
    raw = pd.read_csv(f"./{filename}.csv", dtype=str, keep_default_na=False, encoding='gbk')
except UnicodeDecodeError:
    raw = pd.read_csv(f"./{filename}.csv", dtype=str, keep_default_na=False, encoding='gb18030')


# ================= 列名预处理 =================
def process_colname(col):
    """处理列名中的特殊字符"""
    return re.sub(r'[._]', '0', col)


original_columns = raw.columns.tolist()
raw.columns = [process_colname(col) for col in original_columns]


# ================= 字段类型检测 =================
def detect_column_type(series):
    cleaned = series.str.strip()
    if cleaned.empty: return 'character'
    if cleaned.str.match(r'^[+-]?\d+$').all(): return 'integer'
    try:
        temp = cleaned.str.replace(',', '', regex=False)
        pd.to_numeric(temp, errors='raise')
        return 'numeric' if temp.str.contains(r'\.|e|E').any() else 'integer'
    except:
        return 'character'


attribute = pd.DataFrame({
    'original_col': original_columns,
    'processed_col': raw.columns,
    'type': [detect_column_type(raw[col]) for col in raw.columns],
    'len': [6] * len(raw.columns)
})

# ================= 列宽计算 =================
for i, col in enumerate(raw.columns):
    max_len = raw[col].apply(lambda x: len(str(x).strip())).max()
    attribute.at[i, 'len'] = max(6, max_len)


# ================= 数据对齐 =================
def pad_column(series, width):
    return series.str.strip().apply(lambda x: x.rjust(width))


for col in raw.columns:
    col_len = attribute[attribute['processed_col'] == col]['len'].values[0]
    raw[col] = pad_column(raw[col].astype(str), col_len)

# ================= 生成define.stp =================
define_lines = []
cum_pos = 1
for _, row in attribute.iterrows():
    start = cum_pos
    end = cum_pos + row['len'] - 1
    prefix = {'integer': 'di', 'numeric': 'dw', 'character': 'dc'}[row['type']]
    define_lines.append(f"{prefix} ${row['processed_col']}=${start}-{end},")
    cum_pos = end + 2

# ================= 全新make生成逻辑 =================
multi_choice = defaultdict(int)

for orig_col in original_columns:
    # 跳过字符型字段
    if attribute[attribute['original_col'] == orig_col]['type'].values[0] == 'character':
        continue

    # 拆分基题和子题编号
    if '_' in orig_col:
        parts = orig_col.rsplit('_', 1)  # 从右边第一个下划线分割
        base_part, num_part = parts[0], parts[1]
    else:
        continue  # 仅处理带下划线的字段

    # 清洗基题名
    clean_base = re.sub(r'[._]', '0', base_part)

    # 提取子题编号（自动补零到两位）
    try:
        sub_num = int(num_part)
        sub_num = max(1, min(sub_num, 99))  # 限制1-99
        multi_choice[clean_base] = max(multi_choice[clean_base], sub_num)
    except ValueError:
        continue

# 构建make内容
make_lines = ["[*data ttl(;)=="]
for base in sorted(multi_choice.keys()):
    make_lines.append(f"{base};{multi_choice[base]};")
make_lines += [
    "]",
    "[*do i=1:[ttl.#]/2]",
    "   [*do a=1:[ttl.i*2]]",
    "      om $[ttl.i*2-1]=$[ttl.i*2-1]0[a]/1-999,",
    "   [*end a]",
    "[*end i]"
]

# ================= 文件输出 =================
with open(f"{filename}1.dat", 'w', encoding='gbk') as f:
    for _, row in raw.iterrows():
        f.write(' '.join(row.values) + '\n')

with open(f"{filename}1define.stp", 'w', encoding='gbk') as f:
    f.write('\n'.join(define_lines))

with open(f"{filename}1make.stp", 'w', encoding='gbk') as f:
    if multi_choice:
        f.write('\n'.join(make_lines))
    else:
        f.write("No multiple choice fields detected")

print("处理完成！")
print("识别到的多选题基题：")
print("\n".join([f"{k}: {v}个子题" for k, v in multi_choice.items()]))
print(attribute['type'].value_counts())
print(attribute[attribute['type'] == 'character'])
