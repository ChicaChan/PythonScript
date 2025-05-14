import pandas as pd
import os

def process_question_numbers(file_path):
    """
    处理Excel文件中的题号，将包含两个下划线的题号的第一个下划线替换为点
    
    Args:
        file_path (str): Excel文件路径
    
    Returns:
        pd.DataFrame: 处理后的数据框
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return None
    
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 获取第一列的列名
        first_column = df.columns[0]
        
        # 创建一个新的数据框来存储结果
        result_df = df.copy()
        
        # 遍历第一列的每个值
        for i, value in enumerate(df[first_column]):
            # 检查值是否为字符串类型
            if isinstance(value, str):
                # 计算下划线的数量
                underscore_count = value.count('_')
                
                # 如果包含两个下划线
                if underscore_count == 2:
                    # 找到第一个下划线的位置
                    first_underscore_pos = value.find('_')
                    
                    # 替换第一个下划线为点
                    new_value = value[:first_underscore_pos] + '.' + value[first_underscore_pos+1:]
                    
                    # 更新结果数据框中的值
                    result_df.at[i, first_column] = new_value
        
        return result_df
    
    except Exception as e:
        print(f"处理文件时出错：{str(e)}")
        return None

def main():
    # 设置Excel文件路径
    file_path = "D:\\workplace\\Python脚本\\tmic_tool\\column.xlsx"
    
    # 处理题号
    result_df = process_question_numbers(file_path)
    
    if result_df is not None:
        # 保存处理后的结果到新的Excel文件
        output_file = "column_processed.xlsx"
        result_df.to_excel(output_file, index=False)
        print(f"处理完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    main()