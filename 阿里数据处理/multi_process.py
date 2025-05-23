# -*- coding: utf-8 -*-
"""
多选题数据处理脚本
功能：
1. 识别出列名格式为_加上数字的列，例如q1_1, q1_2, q1_3, q1_4
2. 修改这些列所在的数据，将1替换为列名中_后面的数字，例如q1_3列中的1替换为3
"""

import pandas as pd
import re
import os

filename = 'data'

def process_matrix_data(input_file, output_file=None):
    # 检测文件编码
    with open(input_file, 'rb') as f:
        raw_data = f.read(4096)
    
    try:
        import chardet
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        print(f"检测到文件编码: {encoding}")
    except ImportError:
        encoding = 'utf-8'
        print(f"未安装chardet库，默认使用编码: {encoding}")
    
    try:
        df = pd.read_csv(input_file, encoding=encoding)
    except UnicodeDecodeError:
        for enc in ['gbk', 'gb2312', 'utf-8-sig', 'latin1']:
            try:
                df = pd.read_csv(input_file, encoding=enc)
                encoding = enc
                print(f"使用编码 {enc} 成功读取文件")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise Exception("无法识别文件编码，请手动指定编码")
    
    # 获取列名
    columns = df.columns.tolist()
    
    pattern = re.compile(r'^([a-zA-Z]+\d*)_(\d+)$')
    
    columns_to_process = {}
    
    # 识别符合格式的列名
    for col in columns:
        match = pattern.match(col)
        if match:
            # 提取列名中_后面的数字
            number = int(match.group(2))
            columns_to_process[col] = number
    
    print(f"找到 {len(columns_to_process)} 个符合格式的列")
    
    processed_count = 0
    for col, number in columns_to_process.items():
        # 替换为列名中_后面的数字
        mask = df[col] == 1
        if mask.any():
            df.loc[mask, col] = number
            processed_count += mask.sum()
    
    print(f"共处理了 {processed_count} 个单元格")
    
    if output_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")
        base_name = os.path.basename(os.path.splitext(input_file)[0])
        output_file = os.path.join(output_dir, f"{base_name}_processed.csv")
    
    for col in columns_to_process.keys():
        # 只转换包含数值的列
        if pd.api.types.is_numeric_dtype(df[col]):
            # 将NaN值保留为NaN，其他值转为整数
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else x)
    
    df.to_csv(output_file, index=False, encoding=encoding)
    print(f"处理完成，已保存到: {output_file}")
    
    return df

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "input", f"{filename}.csv")
    output_file = os.path.join(script_dir, "output", f"{filename}_processed.csv")
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"文件 {input_file} 不存在，请确保文件在正确的路径下")
        return
    
    # 处理数据
    process_matrix_data(input_file, output_file)
    print("脚本执行完毕")

if __name__ == "__main__":
    main()