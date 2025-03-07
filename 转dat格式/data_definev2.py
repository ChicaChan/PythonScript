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
CHINESE_PATTERN = re.compile(r'^[\u4e00-\u9fff，。！？、；：“”‘’（）《》【】]+$')


def detect_encoding(file_path: str) -> str:
    """检测文件编码"""
    for encoding in ENCODINGS:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1024)
            return encoding
        except UnicodeDecodeError:
            continue
    return ENCODINGS[0]


def is_all_chinese(series: pd.Series) -> bool:
    """检查Series中的所有非空值是否全为中文"""
    if series.empty:
        return False
    cleaned = series.str.strip().replace(r'^\s*$', pd.NA, regex=True).dropna()
    return not cleaned.empty and cleaned.str.contains(CHINESE_PATTERN, na=False).all()


def detect_column_type(series: pd.Series) -> str:
    """字段类型检测"""
    if series.empty:
        return 'integer'

    cleaned = series.str.strip().replace('', pd.NA).dropna()
    if cleaned.empty:
        return 'integer'

    if cleaned.str.fullmatch(r'^[+-]?\d+$').all():
        return 'integer'

    temp = cleaned.str.replace(',', '', regex=False)
    try:
        pd.to_numeric(temp, errors='raise')
        return 'numeric' if temp.str.contains(r'[.eE]').any() else 'integer'
    except ValueError:
        return 'character'


class DataProcessor:
    def __init__(self, file_path: str, output_dir: str):
        self.file_path = file_path
        self.output_dir = output_dir
        self.raw_df = None
        self.attribute_df = None

        pd.set_option('display.max_columns', None)
        os.chdir(os.path.dirname(file_path))

    def filter_chinese_columns(self) -> None:
        """删除全中文列"""
        cols_to_drop = [col for col in self.raw_df.columns if is_all_chinese(self.raw_df[col])]
        self.raw_df.drop(columns=cols_to_drop, inplace=True)

    def load_data(self) -> None:
        """数据加载"""
        encoding = detect_encoding(self.file_path)
        self.raw_df = pd.read_csv(
            self.file_path,
            dtype=str,
            keep_default_na=False,
            encoding=encoding
        )

    def process_columns(self) -> None:
        """列处理"""
        original_columns = self.raw_df.columns.tolist()
        processed_columns = [COLUMN_CLEAN_PATTERN.sub('0', col) for col in original_columns]
        self.raw_df.columns = processed_columns

        self.attribute_df = pd.DataFrame({
            'original_col': original_columns,
            'processed_col': processed_columns,
            'type': [detect_column_type(self.raw_df[col]) for col in processed_columns]
        })

        def calc_byte_len(s):
            try:
                return len(s.encode('gbk'))
            except:
                return len(str(s))

        lengths = self.raw_df.apply(
            lambda col: col.str.strip().apply(calc_byte_len).max()
        ).fillna(0).astype(int)
        self.attribute_df['len'] = lengths.clip(lower=DEFAULT_LEN).values

    def generate_define(self) -> List[str]:
        """生成排序后的define.stp"""
        lines = []
        current_pos = 1

        # 分离多选题字段和其他字段
        multi_records = []
        normal_records = []

        # 收集多选题字段并排序
        multi_groups = defaultdict(list)
        for _, row in self.attribute_df.iterrows():
            orig_col = row['original_col']
            if '_' in orig_col and row['type'] != 'character':
                parts = orig_col.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    base_part, num_part = parts
                    multi_groups[base_part].append((int(num_part), row))

        # 对每个组进行排序并生成新编号
        for base in multi_groups:
            sorted_group = sorted(multi_groups[base], key=lambda x: x[0])
            for new_idx, (orig_num, row) in enumerate(sorted_group, 1):
                new_proc_col = f"{COLUMN_CLEAN_PATTERN.sub('0', base)}{new_idx:02d}"
                multi_records.append((new_proc_col, row['len'], row['type']))

        # 收集普通字段
        for _, row in self.attribute_df.iterrows():
            orig_col = row['original_col']
            if '_' not in orig_col or not orig_col.split('_')[-1].isdigit() or row['type'] == 'character':
                normal_records.append((row['processed_col'], row['len'], row['type']))

        # 合并字段并生成定义
        for proc_col, length, col_type in multi_records + normal_records:
            end_pos = current_pos + length - 1
            lines.append(f"{COLUMN_TYPES[col_type]} ${proc_col}=${current_pos}-{end_pos},")
            current_pos = end_pos + 2

        return lines

    def align_data(self) -> None:
        """数据对齐处理"""
        for col in self.raw_df.columns:
            width = self.attribute_df.loc[
                self.attribute_df['processed_col'] == col, 'len'
            ].values[0]

            self.raw_df[col] = self.raw_df[col].apply(
                lambda x: self._gbk_rjust(str(x).strip(), width)
            )

    def _gbk_rjust(self, text: str, width: int) -> str:
        current_len = len(text.encode('gbk', errors='ignore'))
        return text if current_len >= width else ' ' * (width - current_len) + text

    def save_files(self) -> None:
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        os.makedirs(self.output_dir, exist_ok=True)

        # 保存dat文件
        dat_path = os.path.join(self.output_dir, f"{base_name}.dat")
        with open(dat_path, 'w', encoding='gbk') as f:
            for _, row in self.raw_df.iterrows():
                f.write(' '.join(row.astype(str)) + '\n')

        # 保存define文件
        define_path = os.path.join(self.output_dir, "define.stp")
        with open(define_path, 'w', encoding='gbk') as f:
            f.write('\n'.join(self.generate_define()))

    def run(self) -> None:
        self.load_data()
        self.filter_chinese_columns()
        self.process_columns()
        self.align_data()
        self.save_files()

        print("字段类型统计:")
        print(self.attribute_df['type'].value_counts())
        print("\n字符型字段列表:")
        print(self.attribute_df[self.attribute_df['type'] == 'character'])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python data_define.py <input_csv> <output_dir>")
        sys.exit(1)

    processor = DataProcessor(sys.argv[1], sys.argv[2])
    processor.run()
