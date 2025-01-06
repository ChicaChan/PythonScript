# -*- coding: utf-8 -*-
import os
import pandas as pd


def read_and_round_csv(file_path, decimal_places=0):
    df = pd.read_csv(file_path, encoding='gbk')

    # 将数值型字段四舍五入到整数
    df[df.select_dtypes(include=['number']).columns] = df.select_dtypes(include=['number']).round(decimal_places)

    return df


def replace_column_dots(df):
    df.columns = df.columns.str.replace(".", "0")
    return df


def determine_column_types(df):
    attribute = pd.DataFrame({'colname': df.columns, 'type': 'character'})
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    attribute.loc[attribute['colname'].isin(numeric_columns), 'type'] = 'numeric'
    return attribute


def replace_missing_values(df, fill_value=" "):
    return df.fillna(fill_value)


def create_fixed_width_data(df):
    # 使用字符串表示的所有字段来计算最大长度
    max_lengths = df.astype(str).map(len).max()
    max_lengths = max_lengths.clip(lower=6)

    # 将所有字段转换为字符串，并应用str.ljust
    df = df.astype(str).apply(lambda col: col.str.ljust(max_lengths[col.name]))
    return df, max_lengths


def create_define_steps(df, max_lengths, attribute):
    starts = [sum(max_lengths.iloc[:i]) + i + 1 for i in range(len(df.columns))]
    ends = [starts[i] + max_lengths.iloc[i] - 1 for i in range(len(df.columns))]

    define_steps = []
    for col, start, end, col_type in zip(df.columns, starts, ends, attribute['type']):
        if col_type == 'character':
            define_steps.append(f"dc ${col}=${start}-{end},")
        else:
            define_steps.append(f"di ${col}=${start}-{end},")
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
    df = read_and_round_csv(f"./{filename}.csv", decimal_places=0)

    # 获取以"R#"开头的列名
    r_columns = df.columns[df.columns.str.startswith('R#')]

    df = df.drop(columns=r_columns)

    df = replace_column_dots(df)
    attribute = determine_column_types(df)
    df = replace_missing_values(df)
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

    # 将 DataFrame 中的 NaN 值替换为整数 0
    df = df.fillna(0)

    # 将 DataFrame 中的数值列转换为整数
    df[df.select_dtypes(include=['number']).columns] = df.select_dtypes(include=['number']).astype(int)

    # 将 DataFrame 转换为固定宽度的字符串格式
    fixed_width_data = df.apply(lambda x: x.str.ljust(max_lengths[x.name]), axis=0)

    # 输出为 .dat 文件
    with open(f"{filename}.dat", 'w') as f:
        for index, row in fixed_width_data.iterrows():
            f.write(''.join(row) + '\n')

    # 打印出所有以dc开头的字段
    print_dc_fields(define_file_path)


if __name__ == "__main__":
    os.chdir("D:\\办公软件\\DP小工具\\2.转dat格式")
    filename = "jdcdata"
    main()
