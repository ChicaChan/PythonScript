import os
import pandas as pd
import re
from collections import defaultdict
from typing import List
import sys

# 常量定义
ENCODINGS = ('gbk', 'gb18030', 'utf-8')
COLUMN_CLEAN_PATTERN = re.compile(r'[._]')
DEFAULT_LEN = 6
COLUMN_TYPES = {
    'integer': 'di',
    'numeric': 'dw',
    'character': 'dc'
}
PD_VERSION = pd.__version__.split('.')[0]


def detect_encoding(file_path: str) -> str:
    """检测文件编码"""
    for encoding in ENCODINGS:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1024)  # 尝试读取前1KB内容
            return encoding
        except UnicodeDecodeError:
            continue
    return ENCODINGS[0]  # 默认返回第一个编码


def detect_column_type(series: pd.Series) -> str:
    """检测数据类型"""
    if series.empty:
        return 'integer'
    cleaned = series.str.strip().replace('', pd.NA).dropna()
    if cleaned.empty:
        return 'integer'
    # 先尝试匹配纯整数格式
    if cleaned.str.fullmatch(r'^[+-]?\d+$').all():
        return 'integer'
    # 处理含逗号的数值格式
    temp = cleaned.str.replace(',', '', regex=False)
    try:
        pd.to_numeric(temp, errors='raise')
        return 'numeric' if temp.str.contains(r'[.eE]').any() else 'integer'
    except ValueError:
        return 'character'


def post_process_define(define_path: str) -> None:
    pattern_multi = re.compile(r'^(di\s+\$)([A-Z]+)(\d+)(0)(\d+)(=\$.*)$')

    with open(define_path, 'r', encoding='gbk') as f:
        lines = f.readlines()

    multi_groups = defaultdict(list)
    for idx, line in enumerate(lines):
        m = pattern_multi.match(line)
        if m:
            key = m.group(2) + m.group(3)
            multi_groups[key].append((idx, m))

    # 对每组多选题变量重新编号
    for key, items in multi_groups.items():
        if len(items) > 1:
            width = len(items[0][1].group(5))  # 获取原始序号位数
            for order, (idx, m) in enumerate(items, start=1):
                new_suffix = str(order).zfill(width)  # 生成新序号
                lines[idx] = m.group(1) + m.group(2) + m.group(3) + m.group(4) + new_suffix + m.group(6) + "\n"

    with open(define_path, 'w', encoding='gbk') as f:
        f.writelines(lines)


# 主处理类
class DataProcessor:

    def __init__(self, file_path: str, output_dir: str):
        self.file_path = file_path
        self.output_dir = output_dir
        self.raw_df = None
        self.attribute_df = None
        self.multi_choice = defaultdict(int)
        pd.set_option('display.max_columns', None)
        os.chdir(os.path.dirname(file_path))

    def load_data(self) -> None:
        """加载CSV文件"""
        encoding = detect_encoding(self.file_path)
        self.raw_df = pd.read_csv(self.file_path, dtype=str, keep_default_na=False, encoding=encoding)

    def process_columns(self) -> None:
        """列处理"""
        original_columns = self.raw_df.columns.tolist()
        # 清洗列名中的特殊字符
        processed_columns = [COLUMN_CLEAN_PATTERN.sub('0', col) for col in original_columns]
        self.raw_df.columns = processed_columns

        # 构建字段属性表
        self.attribute_df = pd.DataFrame({
            'original_col': original_columns,
            'processed_col': processed_columns,
            'type': [detect_column_type(self.raw_df[col]) for col in processed_columns]
        })

        # 计算字段最大字节长度
        def calc_byte_len(s):
            try:
                return len(s.encode('gbk'))
            except Exception:
                return len(str(s))

        lengths = self.raw_df.apply(lambda col: col.str.strip().apply(calc_byte_len).max()).fillna(0).astype(int)
        self.attribute_df['len'] = lengths.clip(lower=DEFAULT_LEN).values

    def align_data(self) -> None:
        """数据对齐处理"""
        for col in self.raw_df.columns:
            width = self.attribute_df.loc[self.attribute_df['processed_col'] == col, 'len'].values[0]
            self.raw_df[col] = self.raw_df[col].apply(lambda x: self._gbk_rjust(str(x).strip(), width))

    def _gbk_rjust(self, text: str, width: int) -> str:
        """GBK编码感知"""
        current_len = len(text.encode('gbk', errors='ignore'))
        if current_len >= width:
            return text
        return ' ' * (width - current_len) + text

    def analyze_multi_choice(self) -> None:
        """多选题分析"""
        for orig_col in self.attribute_df['original_col']:
            col_type = self.attribute_df.loc[self.attribute_df['original_col'] == orig_col, 'type'].values[0]
            if col_type == 'character':
                continue
            if '_' in orig_col:
                base_part, _, num_part = orig_col.rpartition('_')
                if not num_part.isdigit():
                    continue
                sub_num = int(num_part)
                if 1 <= sub_num <= 99:
                    clean_base = COLUMN_CLEAN_PATTERN.sub('0', base_part)
                    self.multi_choice[clean_base] = max(self.multi_choice[clean_base], sub_num)

    def generate_define(self) -> List[str]:
        """生成define文件"""
        lines = []
        current_pos = 1
        for _, row in self.attribute_df.iterrows():
            end_pos = current_pos + row['len'] - 1
            prefix = COLUMN_TYPES[row['type']]
            lines.append(f"{prefix} ${row['processed_col']}=${current_pos}-{end_pos},")
            current_pos = end_pos + 2
        return lines

    def generate_make(self) -> List[str]:
        """生成make文件"""
        if not self.multi_choice:
            return ["No multiple choice fields detected"]
        lines = ["[*data ttl(;)="]
        for base in sorted(self.multi_choice.keys()):
            lines.append(f"{base};{self.multi_choice[base]};")
        lines += [
            "]",
            "[*do i=1:[ttl.#]/2]",
            "   [*do a=1:[ttl.i*2]]",
            "      om $[ttl.i*2-1]=$[ttl.i*2-1]0[a]/1-999,",
            "   [*end a]",
            "[*end i]"
        ]
        return lines

    def save_files(self) -> None:
        """保存文件"""
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        os.makedirs(self.output_dir, exist_ok=True)
        dat_path = os.path.join(self.output_dir, f"{base_name}.dat")
        with open(dat_path, 'w', encoding='gbk') as f:
            for _, row in self.raw_df.iterrows():
                line = ' '.join(row.astype(str)) + '\n'
                f.write(line)
        define_path = os.path.join(self.output_dir, "define.stp")
        with open(define_path, 'w', encoding='gbk') as f:
            f.write('\n'.join(self.generate_define()))
        post_process_define(define_path)
        make_path = os.path.join(self.output_dir, "make.stp")
        with open(make_path, 'w', encoding='gbk') as f:
            f.write('\n'.join(self.generate_make()))

    def run(self) -> None:
        self.load_data()
        self.process_columns()
        self.align_data()
        self.analyze_multi_choice()
        self.save_files()
        print("字段类型统计:")
        print(self.attribute_df['type'].value_counts())
        print("\n字符型字段列表:")
        print(self.attribute_df[self.attribute_df['type'] == 'character'])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python data_definev2.py <input_csv> <output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    processor = DataProcessor(input_file, output_dir)
    processor.run()
    processor.run()
