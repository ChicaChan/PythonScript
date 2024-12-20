# -*- coding: utf-8 -*-
import os
import pandas as pd


def read_and_round_csv(file_path):
    df = pd.read_csv(file_path, encoding='gbk')

    # 将所有数值型字段四舍五入
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].round(0)  # 四舍五入
    return df


def replace_column_dots(df):
    df.columns = df.columns.str.replace(".", "0")
    return df


def determine_column_types(df):
    attribute = pd.DataFrame({'colname': df.columns, 'type': 'character'})
    numeric_columns = df.select_dtypes(include=['int64']).columns
    attribute.loc[attribute['colname'].isin(numeric_columns), 'type'] = 'integer'

    # 检查浮点数列并标记为numeric
    float_columns = df.select_dtypes(include=['float64']).columns
    attribute.loc[attribute['colname'].isin(float_columns), 'type'] = 'numeric'

    return attribute


def create_fixed_width_data(df):
    # 使用字符串表示的所有字段来计算最大长度
    max_lengths = df.astype(str).apply(lambda x: x.str.len()).max()
    max_lengths = max_lengths.clip(lower=6)  # 确保最小长度为6
    return df.astype(str), max_lengths


def create_define_steps(df, max_lengths, attribute):
    starts = [sum(max_lengths.iloc[:i]) + i + 1 for i in range(len(df.columns))]
    ends = [starts[i] + max_lengths.iloc[i] - 1 for i in range(len(df.columns))]

    define_steps = []
    for col, start, end, col_type in zip(df.columns, starts, ends, attribute['type']):
        if col_type == 'character':
            define_steps.append(f"dc ${col}=${start}-${end},")
        elif col_type == 'integer':
            define_steps.append(f"di ${col}=${start}-${end},")
        elif col_type == 'numeric':
            define_steps.append(f"dw ${col}=${start}-${end},")
    return define_steps


def create_make_steps(df):
    col_prefixes = df.columns.str.split("_").str[0]
    col_counts = col_prefixes.value_counts()
    multi_choice_cols = col_counts[col_counts > 1]

    # 过滤掉以"R#"开头的列
    filtered_multi_choice_cols = multi_choice_cols[~multi_choice_cols.index.str.match(r'^R#')]

    if filtered_multi_choice_cols.empty:
        return "No multiple choice, please check your input data."

    make_steps = [
        "[*data ttl(;)=",
        *[f"{col};{count};" for col, count in filtered_multi_choice_cols.items()],
        "]",
        "[*do i=1:[ttl.#]/2]",
        "   [*do a=1:[ttl.i*2]]",
        "      om $[ttl.i*2-1]=$[ttl.i*2-1]0[a]/1-999,",
        "   [*end a]",
        "[*end i]"
    ]
    return make_steps


def print_dc_fields(define_file_path):
    with open(define_file_path, 'r') as f:
        lines = f.readlines()

    dc_fields = [line.strip() for line in lines if line.startswith('dc')]
    for field in dc_fields:
        print(field)


def main():
    df = read_and_round_csv(f"./{filename}.csv")

    # 获取以"R#"开头的列名
    r_columns = df.columns[df.columns.str.startswith('R#')]
    df = df.drop(columns=r_columns)

    df = replace_column_dots(df)
    attribute = determine_column_types(df)

    df, max_lengths = create_fixed_width_data(df)
    define_steps = create_define_steps(df, max_lengths, attribute)
    make_steps = create_make_steps(df)

    define_steps = [step.replace('_', '0') for step in define_steps]

    define_file_path = f"{filename}define.stp"
    with open(define_file_path, 'w') as f:
        for step in define_steps:
            f.write(step + '\n')

    with open(f"{filename}make.stp", 'w') as f:
        for step in make_steps:
            f.write(step + '\n')

    # 将空值替换为 NaN
    df = df.where(pd.notnull(df), None)

    # 输出为 .dat 文件，确保每列之间有空格分隔
    with open(f"{filename}1.dat", 'w') as f:
        for index, row in df.iterrows():
            # 处理缺失值，保持为空字符串
            row_data = [value if value is not None else 'nan' for value in row]  # 保持原值

            # 确保每列的宽度至少为 6
            formatted_row = []
            for i, value in enumerate(row_data):
                width = max(6, max_lengths[i])  # 确保宽度至少为 6
                formatted_row.append(value.rjust(width))  # 使用 rjust 填充宽度

            f.write(''.join(formatted_row).replace('nan', ' ') + '\n')  # 使用空字符串连接每列

    # 后处理 .dat 文件，确保每列宽度至少为 6
    post_process_dat_file(f"{filename}1.dat", max_lengths)

    # 打印出所有以dc开头的字段
    print_dc_fields(define_file_path)


def post_process_dat_file(file_path, max_lengths):
    # 读取文件内容
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 处理每一行
    with open(file_path, 'w') as f:
        for line in lines:
            # 去除 'nan' 字符串
            line = line.replace('nan', '')
            # 将一位小数转换为整数
            new_line = []
            for x in line.split():
                try:
                    # 尝试将字符串转换为浮点数
                    num = float(x)
                    # 检查是否为一位小数
                    if num.is_integer():
                        new_line.append(str(int(num)))  # 转换为整数
                    else:
                        new_line.append(x)  # 保持原值
                except ValueError:
                    new_line.append(x)  # 如果无法转换，保持原值

            # 确保每列的宽度至少为 6
            formatted_row = []
            for i, value in enumerate(new_line):
                # 使用 max_lengths 来确定每列的宽度
                width = max(6, max_lengths[i])  # 确保宽度至少为 6
                formatted_row.append(value.rjust(width))  # 使用 rjust 填充宽度

            f.write(''.join(formatted_row) + '\n')


if __name__ == "__main__":
    os.chdir("D:\\办公软件\\DP小工具\\2.转dat格式")
    filename = "jdcdata"
    main()
