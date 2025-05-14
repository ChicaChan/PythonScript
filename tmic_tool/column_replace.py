import pandas as pd
import os
import re

def replace_column_names(input_file, output_file=None):
    """
    根据映射表替换 Excel 文件中的列名
    
    参数:
        input_file: 输入的 Excel 文件路径
        output_file: 输出的 Excel 文件路径，如果为 None，则覆盖原文件
    """
    if output_file is None:
        # 如果没有指定输出文件，创建一个临时文件名
        file_dir = os.path.dirname(input_file)
        file_name = os.path.basename(input_file)
        name, ext = os.path.splitext(file_name)
        output_file = os.path.join(file_dir, f"{name}_replaced{ext}")
    
    # 读取源数据（第一个 Sheet）
    source_data = pd.read_excel(input_file, sheet_name=0)
    
    # 读取映射表（第二个 Sheet）
    mapping_table = pd.read_excel(input_file, sheet_name=1)
    
    # 创建映射字典
    mapping_dict = dict(zip(mapping_table.iloc[:, 0], mapping_table.iloc[:, 1]))
    
    # 获取原始列名
    original_columns = source_data.columns.tolist()
    new_columns = []
    
    # 替换列名
    for col in original_columns:
        # 检查列名是否包含下划线
        if '_' in col:
            # 分割列名，例如 "q1_1" 分割为 "q1" 和 "1"
            prefix, suffix = col.split('_', 1)
            
            # 检查前缀是否在映射表中
            if prefix in mapping_dict:
                # 替换前缀
                new_col = f"{mapping_dict[prefix]}_{suffix}"
                new_columns.append(new_col)
            else:
                # 如果前缀不在映射表中，保持原列名
                new_columns.append(col)
        else:
            # 检查整个列名是否在映射表中
            if col in mapping_dict:
                new_columns.append(mapping_dict[col])
            else:
                new_columns.append(col)
    
    # 重命名列
    source_data.columns = new_columns
    
    # 保存结果
    source_data.to_excel(output_file, index=False)
    
    return output_file

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        result_file = replace_column_names(input_file, output_file)
        print(f"列名替换完成，结果已保存至: {result_file}")
    else:
        print("使用方法: python column_replace.py 输入文件.xlsx [输出文件.xlsx]")
        
        # 交互式输入
        input_file = input("请输入 Excel 文件路径: ")
        if input_file:
            output_file = input("请输入输出文件路径 (留空则自动生成): ")
            output_file = output_file if output_file else None
            
            result_file = replace_column_names(input_file, output_file)
            print(f"列名替换完成，结果已保存至: {result_file}")