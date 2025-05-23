import pandas as pd
import os
import glob
import argparse
from colorama import init, Fore, Back, Style
from tabulate import tabulate

# 初始化colorama
init(autoreset=True)

def combine_excel_files(show_column_analysis=True, missing_threshold=10):
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义输入和输出文件夹的路径
    input_folder = os.path.join(script_dir, 'input')
    output_folder = os.path.join(script_dir, 'output')
    output_file_path = os.path.join(output_folder, 'combined_data.xlsx')

    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"{Fore.RED}错误：输入文件夹 '{input_folder}' 不存在。请创建该文件夹并将Excel文件放入其中。{Style.RESET_ALL}")
        return

    # 创建输出文件夹（如果它不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"{Fore.GREEN}✓ 已创建输出文件夹：'{output_folder}'{Style.RESET_ALL}")

    # 查找输入文件夹中所有的Excel文件（.xlsx 和 .xls）
    excel_files = glob.glob(os.path.join(input_folder, '*.xlsx')) + glob.glob(os.path.join(input_folder, '*.xls'))

    if not excel_files:
        print(f"{Fore.RED}在输入文件夹 '{input_folder}' 中没有找到Excel文件。{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}找到以下Excel文件进行合并：{Style.RESET_ALL}")
    for file in excel_files:
        print(f"  {Fore.YELLOW}• {os.path.basename(file)}{Style.RESET_ALL}")

    all_data = []

    for file in excel_files:
        try:
            df = pd.read_excel(file, engine='openpyxl' if file.endswith('.xlsx') else None)
            all_data.append(df)
            print(f"{Fore.GREEN}✓ 已读取文件：'{os.path.basename(file)}'{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}读取文件 '{os.path.basename(file)}' 时出错：{e}{Style.RESET_ALL}")
            continue

    if not all_data:
        print(f"{Fore.RED}没有成功读取任何Excel文件的数据。{Style.RESET_ALL}")
        return

    # 纵向合并所有数据框（基于列名）
    # ignore_index=True 会重新生成索引
    # sort=False 会保持原始列的顺序，如果列不完全匹配，则会引入NaN
    try:
        # 在合并前收集所有文件的列名
        all_columns = set()
        file_columns = {}
        
        for i, df in enumerate(all_data):
            file_name = os.path.basename(excel_files[i])
            file_columns[file_name] = set(df.columns)
            all_columns.update(df.columns)
            
        combined_df = pd.concat(all_data, ignore_index=True, sort=False)
        print(f"{Fore.GREEN}✓ {Style.BRIGHT}数据合并完成。{Style.RESET_ALL}")
        
        # 显示合并后的数据基本信息
        if not combined_df.empty:
            print(f"\n{Fore.CYAN}{Style.BRIGHT}合并后的数据共有 {len(combined_df)} 行，{len(combined_df.columns)} 列{Style.RESET_ALL}")
        
        # 分析每个文件中缺失的列（如果启用了列名分析）
        if show_column_analysis:
            print(f"\n{Fore.YELLOW}{Style.BRIGHT}各文件列名分析：{Style.RESET_ALL}")
            
            # 计算每个文件缺失的列数
            files_with_missing = 0
            # 收集所有缺失列的信息，用于生成缺失列报告
            missing_columns_data = []
            
            for file_name, columns in file_columns.items():
                missing_columns = all_columns - columns
                if missing_columns:
                    files_with_missing += 1
                    # 为每个缺失的列添加一条记录
                    for col in missing_columns:
                        missing_columns_data.append({
                            '文件名': file_name,
                            '缺失列': col
                        })
            
            if files_with_missing > 0:
                print(f"{Fore.YELLOW}发现 {files_with_missing} 个文件存在列缺失情况{Style.RESET_ALL}")
                
                # 创建一个标志，表示是否有文件缺失列较多
                has_many_missing_columns = False
                
                for file_name, columns in file_columns.items():
                    missing_columns = all_columns - columns
                    if missing_columns:
                        missing_count = len(missing_columns)
                        print(f"{Fore.RED}● {file_name} 缺少 {missing_count} 列{Style.RESET_ALL}")
                        
                        # 如果缺失的列太多，只显示数量而不显示详细列表
                        if missing_count <= missing_threshold:
                            print(f"  {', '.join(sorted(missing_columns))}")
                        else:
                            print(f"  缺失的列数量较多，共 {missing_count} 列")
                            has_many_missing_columns = True
                    else:
                        print(f"{Fore.GREEN}✓ {file_name} 包含所有列{Style.RESET_ALL}")
                
                # 如果有文件缺失列较多，生成一个额外的Excel文件
                if has_many_missing_columns and missing_columns_data:
                    missing_columns_df = pd.DataFrame(missing_columns_data)
                    missing_columns_output_path = os.path.join(output_folder, 'missing_columns_report.xlsx')
                    try:
                        missing_columns_df.to_excel(missing_columns_output_path, index=False, engine='openpyxl')
                        print(f"\n{Fore.CYAN}{Style.BRIGHT}✓ 由于存在大量缺失列，已生成缺失列详细报告：'{missing_columns_output_path}'{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}保存缺失列报告时出错：{e}{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}合并数据时出错：{e}{Style.RESET_ALL}")
        return

    # 将合并后的数据保存到新的Excel文件
    try:
        combined_df.to_excel(output_file_path, index=False, engine='openpyxl')
        print(f"\n{Fore.GREEN}{Style.BRIGHT}✓ 合并后的数据已成功保存到：'{output_file_path}'{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}保存合并后的文件时出错：{e}{Style.RESET_ALL}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='合并多个Excel文件')
    parser.add_argument('--no-analysis', action='store_true', help='不显示列名分析信息')
    parser.add_argument('--missing-threshold', type=int, default=10, help='设置缺失列数量的阈值，超过此值将被视为缺失列较多并生成报告（默认：10）')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    combine_excel_files(show_column_analysis=not args.no_analysis, missing_threshold=args.missing_threshold)