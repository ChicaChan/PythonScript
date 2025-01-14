# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np

# 设置工作目录
os.chdir("D:\\DP小工具\\2.转dat格式")

# 文件名
filename = "data"

# 读取CSV文件
try:
    # 首先尝试用UTF-8读取
    raw = pd.read_csv(f"./{filename}.csv")
except UnicodeDecodeError:
    try:
        # 如果UTF-8失败，尝试用GBK读取
        raw = pd.read_csv(f"./{filename}.csv", encoding='GBK')
    except UnicodeDecodeError:
        # 如果GBK也失败，尝试用GB2312读取
        raw = pd.read_csv(f"./{filename}.csv", encoding='GB2312')

# 修改数值列处理
numeric_cols = raw.select_dtypes(include=[np.number]).columns
# 对数值列进行四舍五入，但保持NA值
for col in numeric_cols:
    # 只对非NA值进行四舍五入和整数转换
    mask = raw[col].notna()
    raw.loc[mask, col] = raw.loc[mask, col].round().astype(int)

# 将列名中的'.'替换为'0'
raw.columns = raw.columns.str.replace(".", "0")

# 创建属性数据框
attribute = pd.DataFrame({
    'colname': raw.columns,
    'type': 0
})
attribute['colname'] = attribute['colname'].str.replace("_", "0")

# 确定每列的数据类型
for i, col in enumerate(raw.columns):
    if raw[col].dtype == 'object':
        attribute.loc[i, 'type'] = 'character'
    elif raw[col].dtype in ['float64', 'float32']:
        attribute.loc[i, 'type'] = 'numeric'
    else:
        attribute.loc[i, 'type'] = 'integer'

# 将NA值替换为多个空格
raw = raw.fillna(" " * 6)  # 使用6个空格填充NA值

# 计算每列的最大字符长度
def get_max_bytes_length(series):
    return max(len(str(x).encode()) for x in series)

len_list = [get_max_bytes_length(raw[col]) for col in raw.columns]
len_list = [max(x, 6) for x in len_list]
attribute['len'] = len_list

# 定义填充函数
def fix_data(series):
    max_len = get_max_bytes_length(series)
    max_len = max(max_len, 6)
    # 对于NA值保持为空格，对于数值进行格式化
    return series.apply(lambda x: (" " * (max_len - len(str(x).encode()))) + str(x))

# 对每列进行填充
for col in raw.columns:
    raw[col] = fix_data(raw[col])

# 计算define.stp
s = [1]
for i in range(1, len(attribute)):
    s.append(sum(attribute['len'][:i]) + i)
    
e = [x + y - 1 for x, y in zip(s, attribute['len'])]

# 生成define语句
def get_define_str(row):
    type_map = {
        'integer': 'di',
        'numeric': 'dw',
        'character': 'dc'
    }
    return f"{type_map[row['type']]} ${row['colname']}=${row['s']}-{row['e']},"

attribute['s'] = s
attribute['e'] = e
attribute['define'] = attribute.apply(get_define_str, axis=1)

# 生成make.stp
colname1 = raw.columns
colname2 = [x.split('_')[0] for x in colname1]
colname_counts = pd.Series(colname2).value_counts()
m_colname = colname_counts[colname_counts > 1]

if len(m_colname) == 0:
    make = ["No multiple choice,please check your input data."]
else:
    make = ["[*data ttl(;)="]
    make.extend([f"{idx};{val};" for idx, val in m_colname.items()])
    make.extend([
        "]",
        "[*do i=1:[ttl.#]/2]",
        "   [*do a=1:[ttl.i*2]]",
        "      om $[ttl.i*2-1]=$[ttl.i*2-1]0[a]/1-999,",
        "   [*end a]",
        "[*end i]"
    ])

# 输出文件
with open(f"{filename}.dat", 'w', encoding='utf-8') as f:
    for _, row in raw.iterrows():
        f.write(' '.join(str(x) for x in row) + '\n')

with open(f"{filename}define.stp", 'w', encoding='utf-8') as f:
    f.write('\n'.join(attribute['define']))

with open(f"{filename}make.stp", 'w', encoding='utf-8') as f:
    f.write('\n'.join(make))

# 打印文本型字段
print(attribute[attribute['type'] == 'character']) 