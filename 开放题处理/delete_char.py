import csv
import re

input_file = 'd:/workplace/Python脚本/开放题处理/input.csv'
output_file = 'd:/workplace/Python脚本/开放题处理/output.csv'

def clean_cell(cell):
    # 保留数字，去除其他字符
    return re.sub(r'[^\d]', '', cell)

with open(input_file, 'r', encoding='gbk') as infile, \
     open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for idx, row in enumerate(reader):
        if idx == 0:
            # 首行列名不变
            writer.writerow(row)
        else:
            new_row = [clean_cell(cell) for cell in row]
            writer.writerow(new_row)