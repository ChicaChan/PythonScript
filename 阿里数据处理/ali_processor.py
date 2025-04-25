# -*- coding: utf-8 -*-
import pandas as pd
import re
import os

filename = '示例'

def process_multiple_choice(df):
    """
    处理多选题合并
    """
    # 创建DataFrame副本
    df = df.copy()
    
    columns = df.columns.tolist()
    multi_choice_bases = set()
    new_columns = {}

    # 识别多选题基名
    for col in columns:
        if re.search(r'_\d+$', col):
            base = re.sub(r'_\d+$', '', col)
            multi_choice_bases.add(base)

    # 预处理所有多选题
    for base in multi_choice_bases:
        pattern = fr'^{re.escape(base)}_\d+$'
        related_cols = [col for col in columns if re.match(pattern, col)]

        if related_cols:
            # 处理所有行的数据
            new_columns[base] = df[related_cols].apply(
                lambda row: ';'.join([str(re.search(r'_(\d+)$', col).group(1)) 
                                     for col in related_cols if pd.notna(row[col])]) or pd.NA, 
                axis=1
            )
    
    # 使用pd.concat一次性添加所有新列，而不是循环添加
    if new_columns:
        # 将字典转换为DataFrame
        new_df = pd.DataFrame(new_columns)
        # 使用concat一次性合并
        df = pd.concat([df, new_df], axis=1)
    
    # 删除原列
    columns_to_drop = []
    for base in multi_choice_bases:
        pattern = fr'^{re.escape(base)}_\d+$'
        related_cols = [col for col in columns if re.match(pattern, col)]
        columns_to_drop.extend(related_cols)
    
    df = df.drop(columns=columns_to_drop)

    return df

def process_matrix_data(input_file, output_file=None):
    """
    处理矩阵题数据
    """
    # 检测文件编码
    with open(input_file, 'rb') as f:
        raw_data = f.read(4096)
    
    # 检测编码
    try:
        import chardet
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        print(f"检测到文件编码: {encoding}")
    except ImportError:
        encoding = 'utf-8'
        print(f"未安装chardet库，默认使用编码: {encoding}")
    
    # 读取CSV文件
    try:
        # 添加low_memory=False解决DtypeWarning
        df = pd.read_csv(input_file, encoding=encoding, low_memory=False)
    except UnicodeDecodeError:
        for enc in ['gbk', 'gb2312', 'utf-8-sig', 'latin1']:
            try:
                # 添加low_memory=False解决DtypeWarning
                df = pd.read_csv(input_file, encoding=enc, low_memory=False)
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
        mask = df[col] == 1
        if mask.any():
            df.loc[mask, col] = number
            processed_count += mask.sum()
    
    print(f"共处理了 {processed_count} 个单元格")
    
    if output_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(os.path.splitext(input_file)[0])
        output_file = os.path.join(output_dir, f"{base_name}_processed.csv")
    

    # 将数值列转换为整数
    for col in columns_to_process.keys():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else x)
    
    df.to_csv(output_file, index=False, encoding=encoding)
    
    return df

def combined_processing(input_file, final_output=None):
    # 添加low_memory=False解决DtypeWarning
    matrix_processed = process_matrix_data(input_file)
    
    final_df = process_multiple_choice(matrix_processed)

    # 输出路径
    if not final_output:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")
        base_name = os.path.basename(os.path.splitext(input_file)[0])
        final_output = os.path.join(output_dir, f"{base_name}_output.csv")

    # 编码一致
    encoding = 'utf-8'
    with open(input_file, 'rb') as f:
        try:
            import chardet
            result = chardet.detect(f.read(4096))
            encoding = result['encoding']
        except ImportError:
            pass

    # 保存
    final_df.to_csv(final_output, index=False, encoding=encoding)
    print(f"output文件已保存至: {final_output}")
    return final_df

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "input", f"{filename}.csv")
    output_path = os.path.join(script_dir, "output", f"{filename}_final.csv")
    
    if os.path.exists(input_path):
        combined_processing(input_path, output_path)
    else:
        print("输入文件不存在，请检查路径配置")