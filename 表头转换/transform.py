import openpyxl

def main():
    wb = openpyxl.load_workbook("表头转换/input.xlsx")
    ws = wb.active
    
    hierarchy = []  # 层级栈
    counters = {}    # 路径计数器
    output = []

    for cell in ws[1]:
        value = str(cell.value).strip() if cell.value else ""
        if not value:
            continue

        # 解析层级结构
        parts = [p.strip() for p in value.split('**') if p.strip()]
        current_depth = len(parts)

        # 更新层级栈
        del hierarchy[current_depth-1:]
        hierarchy.extend(parts)

        # 生成完整路径
        full_path = '**'.join(hierarchy)
        
        # 更新计数器
        counters.setdefault(full_path, 0)
        counters[full_path] += 1

        # 构建输出格式
        if len(hierarchy) > 1:
            parent = '**'.join(hierarchy[:-1])
            current = hierarchy[-1]
            output.append(f':"{parent}**{current}*([~{counters[full_path]}])",')
        else:
            output.append(f':"{full_path}*([~{counters[full_path]}])",')

    # 写入结果文件
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))
        
    print("转换完成，结果已保存到 output.txt")

if __name__ == "__main__":
    main()