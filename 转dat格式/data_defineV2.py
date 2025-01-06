import pandas as pd
import numpy as np
import os

# 设置工作目录
os.chdir("D:\\办公软件\\DP小工具\\2.转dat格式")

# 读取数据
filename = "data"
raw = pd.read_csv(f"./{filename}.csv", dtype=str)

# 四舍五入数值列到五位小数
numeric_cols = raw.select_dtypes(include=[np.number]).columns
raw[numeric_cols] = raw[numeric_cols].round(5)

# 将点替换为0
raw.columns = raw.columns.str.replace('.', '0')

# 创建属性数据框
attribute = pd.DataFrame({
    'colname': raw.columns,
    'type': [''] * len(raw.columns),  # 初始化为字符串类型
    'len': [0] * len(raw.columns)  # 初始长度设置为0
})

# 替换下划线为0
attribute['colname'] = attribute['colname'].str.replace('_', '0')

# 确定数据类型
for i, col in enumerate(raw.columns):
    if raw[col].dtype == 'object':
        attribute.at[i, 'type'] = 'character'
    elif raw[col].dtype.kind in 'bifc':  # 包含整数和浮点数
        attribute.at[i, 'type'] = 'numeric'
    else:
        attribute.at[i, 'type'] = 'integer'

# 用空格填充NA值
raw = raw.fillna(' ')

# 计算每列的最大长度
len_max = raw.apply(lambda x: x.str.len()).max()
for i in range(len(raw.columns)):
    attribute.at[i, 'len'] = max(len_max.iloc[i], 6)  # 确保至少6个字符宽

# 用空格填充实现等宽数据
for col in raw.columns:
    col_len_series = attribute[attribute['colname'] == col]['len']
    if not col_len_series.empty:  # 确保条件筛选不为空
        col_len = col_len_series.values[0]  # 获取列的长度
        raw[col] = raw[col].apply(lambda x: str(x).ljust(col_len))

# 定义.stp
s = [0]
for i in range(len(attribute)):
    s.append(s[-1] + attribute.iloc[i]['len'])

# 创建define字段
attribute['define'] = ''
for i in range(len(attribute)):
    col_type = attribute.iloc[i]['type']
    col_name = attribute.iloc[i]['colname']
    start = s[i] + 1  # 起始位置加1
    end = s[i + 1]  # 结束位置
    if col_type == 'integer':
        attribute.at[i, 'define'] = f"di {col_name}=$ {start}-{end},"
    elif col_type == 'numeric':
        attribute.at[i, 'define'] = f"dw {col_name}=$ {start}-{end},"
    elif col_type == 'character':
        attribute.at[i, 'define'] = f"dc {col_name}=$ {start}-{end},"

# 制作make.stp
colname1 = raw.columns
colname2 = colname1.str.replace(r'_.*', '')
colname_num = colname2.value_counts()
m_colname = colname_num[colname_num > 1]

if len(m_colname) == 0:
    make = "No multiple choice, please check your input data."
else:
    make = "[*data ttl(;)=\n"
    for name, freq in m_colname.items():
        make += f"{name[0:3]};{name[3:]};\n"
    make += "],\n[*do i=1:[ttl.#]/2]\n   [*do a=1:[ttl.i*2]]\n      om $[ttl.i*2-1]=$[ttl.i*2-1]0[a]/1-999,\n   [*end a]\n[*end i]"

# 输出结果
raw.to_csv(f"{filename}.dat", index=False, header=False)
attribute.to_csv(f"{filename}define.stp", index=False, header=False)
make_df = pd.DataFrame([make], columns=['df'])
make_df.to_csv(f"{filename}make.stp", index=False, header=False)
