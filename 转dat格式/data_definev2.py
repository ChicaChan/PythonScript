import os
import pandas as pd
import re
from collections import defaultdict
from typing import List, Dict, Tuple
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
CHINESE_PATTERN = re.compile(r'^[\u4e00-\u9fff，。！？、；：“”‘’（）《》【】]+$')  # 新增中文检测正则


# 工具函数
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


def is_all_chinese(series: pd.Series) -> bool:  # 新增中文检测函数
    """检查Series中的所有非空值是否全为中文"""
    if series.empty:
        return False
    cleaned = series.str.strip().replace(r'^\s*$', pd.NA, regex=True).dropna()
    if cleaned.empty:
        return False
    return cleaned.str.contains(CHINESE_PATTERN, na=False).all()


def detect_column_type(series: pd.Series) -> str:
    """字段类型检测"""
    if series.empty:
        return 'integer'

    # 预处理数据
    cleaned = series.str.strip().replace('', pd.NA).dropna()
    if cleaned.empty:
        return 'integer'

    # 整数检测
    if cleaned.str.fullmatch(r'^[+-]?\d+$').all():
        return 'integer'

    # 数值型检测
    temp = cleaned.str.replace(',', '', regex=False)
    try:
        pd.to_numeric(temp, errors='raise')
        return 'numeric' if temp.str.contains(r'[.eE]').any() else 'integer'
    except ValueError:
        return 'character'


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

    def renumber_multi_choice_columns(self) -> None:  # 新增方法
        """重新编号多选题字段名"""
        original_columns = self.raw_df.columns.tolist()
        column_rename_map = {}
        multi_choice_groups = defaultdict(list)

        for col in original_columns:
            if '_' in col:
                parts = col.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    base_part, num_part = parts
                    multi_choice_groups[base_part].append((col, int(num_part)))

        for base_part, cols in multi_choice_groups.items():
            sorted_cols = sorted(cols, key=lambda x: x[1])
            new_number = 1
            for old_col, _ in sorted_cols:
                new_col = f"{base_part}_{new_number}"
                column_rename_map[old_col] = new_col
                new_number += 1

        self.raw_df.rename(columns=column_rename_map, inplace=True)

    def filter_chinese_columns(self) -> None:  # 新增方法
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
        if current_len >= width:
            return text
        return ' ' * (width - current_len) + text

    def analyze_multi_choice(self) -> None:
        for orig_col in self.attribute_df['original_col']:
            col_type = self.attribute_df.loc[
                self.attribute_df['original_col'] == orig_col, 'type'
            ].values[0]
            if col_type == 'character':
                continue

            if '_' in orig_col:
                base_part, _, num_part = orig_col.rpartition('_')
                if not num_part.isdigit():
                    continue

                sub_num = int(num_part)
                if 1 <= sub_num <= 99:
                    clean_base = COLUMN_CLEAN_PATTERN.sub('0', base_part)
                    self.multi_choice[clean_base] = max(
                        self.multi_choice[clean_base],
                        sub_num
                    )

    def generate_define(self) -> List[str]:
        lines = []
        current_pos = 1
        for _, row in self.attribute_df.iterrows():
            end_pos = current_pos + row['len'] - 1
            prefix = COLUMN_TYPES[row['type']]
            lines.append(
                f"{prefix} ${row['processed_col']}=${current_pos}-{end_pos},"
            )
            current_pos = end_pos + 2
        return lines

    def generate_make(self) -> List[str]:
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

        make_path = os.path.join(self.output_dir, "make.stp")
        with open(make_path, 'w', encoding='gbk') as f:
            f.write('\n'.join(self.generate_make()))

    def run(self) -> None:  # 修改运行流程
        self.load_data()
        self.filter_chinese_columns()  # 新增步骤
        self.renumber_multi_choice_columns()  # 新增步骤
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
        print("Usage: python data_define.py <input_csv> <output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    processor = DataProcessor(input_file, output_dir)
    processor.run()
