# -*- coding: utf-8 -*-
import os
import pandas as pd
import re
import csv


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def read_and_round_csv(file_path, decimal_places=5):
    df = pd.read_csv(file_path, dtype=str)
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
    max_lengths = df.astype(str).map(len).max()
    max_lengths = max_lengths.clip(lower=6)
    df = df.apply(lambda col: col.str.ljust(max_lengths[col.name]), axis=0)
    return df, max_lengths


def create_define_steps(df, max_lengths, attribute):
    starts = [sum(max_lengths.iloc[:i]) + i + 1 for i in range(len(df.columns))]
    ends = [starts[i] + max_lengths[i] - 1 for i in range(len(df.columns))]

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
        *[f"{col};{count}" for col, count in filtered_multi_choice_cols.items()],
        "]",
        "[*do i=1:[ttl.#]/2]",
        "   [*do a=1:[ttl.i*2]]",
        "      om $[ttl.i*2-1]=$[ttl.i*2-1]0[a]/1-999,",
        "   [*end a]",
        "[*end i]"
    ]
    return make_steps


def main():
    clear_screen()
    os.chdir("D:\\办公软件\\DP小工具\\2.转dat格式")

    filename = "ditu4"
    df = read_and_round_csv(f"./{filename}.csv")

    # 获取以"R#"开头的列名
    r_columns = df.columns[df.columns.str.startswith('R#')]

    # 使用列名来过滤DataFrame
    df = df.drop(columns=r_columns)

    df = replace_column_dots(df)
    attribute = determine_column_types(df)
    df = replace_missing_values(df)
    df, max_lengths = create_fixed_width_data(df)
    define_steps = create_define_steps(df, max_lengths, attribute)
    make_steps = create_make_steps(df)

    # 删除双引号
    define_steps = [step.replace('"', '') for step in define_steps]

    with open(f"{filename}define.stp", 'w') as f:
        for step in define_steps:
            f.write(step + '\n')

    with open(f"{filename}make.stp", 'w') as f:
        for step in make_steps:
            f.write(step + '\n')

    df.to_csv(f"{filename}.dat", index=False, header=False)
    # pd.DataFrame(make_steps, columns=['make']).to_csv(f"{filename}make.stp", index=False, header=False)

    print(attribute[attribute['type'] == 'character'])


if __name__ == "__main__":
    main()

