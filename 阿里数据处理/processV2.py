import pandas as pd
import re
import sys
import os
import time

def process_excel(input_file):
    try:
        print(f"开始处理文件：{input_file}")
        print("正在读取Excel文件...")
        
        # 读取Excel文件
        df = pd.read_excel(input_file)
        
        # 获取所有列名
        columns = df.columns
        print(f"共发现 {len(columns)} 个列")

        # 找出所有多选题的基础名称
        multi_choice_bases = set()
        for col in columns:
            if re.search(r'_\d+$', col):
                base = re.sub(r'_\d+$', '', col)
                multi_choice_bases.add(base)
        
        print(f"发现 {len(multi_choice_bases)} 个多选题需要处理")

        # 处理每个多选题
        for i, base in enumerate(multi_choice_bases, 1):
            print(f"正在处理第 {i}/{len(multi_choice_bases)} 个多选题：{base}")
            
            # 找出该题目的所有以数字结尾的子列
            pattern = fr'^{re.escape(base)}_\d+$'
            related_cols = [col for col in columns if re.match(pattern, col)]
            
            # 创建新列来存储合并后的结果
            def process_row(row):
                selected = []
                for col in related_cols:
                    number = re.search(r'_(\d+)$', col)
                    if number and pd.notna(row[col]):
                        selected.append(number.group(1))
                return ';'.join(selected) if selected else ''

            # 只有当存在相关的子列时才进行处理
            if related_cols:
                df[base] = df[related_cols].apply(process_row, axis=1)
                df = df.drop(columns=related_cols)

        # 生成输出文件名
        input_dir = os.path.dirname(input_file)
        input_filename = os.path.basename(input_file)
        filename, ext = os.path.splitext(input_filename)
        output_filename = f'processed_{filename}_{time.strftime("%Y%m%d_%H%M%S")}{ext}'
        output_path = os.path.join(input_dir, output_filename)

        # 保存处理后的数据
        print("正在保存处理后的文件...")
        df.to_excel(output_path, index=False)
        
        print(f"\n处理完成！")
        print(f"输出文件已保存至：{output_path}")
        return True

    except Exception as e:
        print(f"\n处理过程中出现错误：")
        print(str(e))
        return False

def main():
    try:
        # 检查是否有文件被拖拽到程序上
        if len(sys.argv) > 1:
            input_file = sys.argv[1]
            if not os.path.exists(input_file):
                print(f"错误：找不到文件 {input_file}")
                input("按回车键退出...")
                return

            success = process_excel(input_file)
            
            if success:
                print("\n程序执行成功！")
            else:
                print("\n程序执行失败！")
        else:
            print("请将Excel文件拖拽到程序上进行处理。")
        
        print("\n按回车键退出...")
        input()

    except Exception as e:
        print(f"\n程序发生未预期的错误：")
        print(str(e))
        print("\n按回车键退出...")
        input()

if __name__ == '__main__':
    main()