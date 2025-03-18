# Python
import os
import pandas as pd
import re
from collections import defaultdict
from typing import List
import sys

# Constants definition
ENCODINGS = ('gbk', 'gb18030', 'utf-8')
COLUMN_CLEAN_PATTERN = re.compile(r'[._]')
DEFAULT_LEN = 6
COLUMN_TYPES = {
    'integer': 'di',
    'numeric': 'dw',
    'character': 'dc'
}
PD_VERSION = pd.__version__.split('.')[0]


# Utility functions

def detect_encoding(file_path: str) -> str:
    """Detect file encoding"""
    for encoding in ENCODINGS:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1024)
            return encoding
        except UnicodeDecodeError:
            continue
    return ENCODINGS[0]


def detect_column_type(series: pd.Series) -> str:
    """Detect column type"""
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


def post_process_define(define_path: str) -> None:
    """
    Post process the define file.
    Only process multiple choice lines, e.g.:
       di $S601=$194-199,
       di $S602=$201-206,
       di $S603=$208-213,
    where the line contains an explicit '0' separator.
    Each group is reassigned to have consecutively numbered sub-questions.
    Single choice lines remain unchanged.
    """
    # Regex for multiple choice: match lines with an explicit '0' separator
    pattern_multi = re.compile(r'^(di\s+\$)([A-Z]+)(\d+)(0)(\d+)(=\$.*)$')

    with open(define_path, 'r', encoding='gbk') as f:
        lines = f.readlines()

    # Process multiple choice lines grouped by letter+base number.
    multi_groups = defaultdict(list)
    for idx, line in enumerate(lines):
        m = pattern_multi.match(line)
        if m:
            key = m.group(2) + m.group(3)
            multi_groups[key].append((idx, m))

    for key, items in multi_groups.items():
        if len(items) > 1:
            width = len(items[0][1].group(5))
            for order, (idx, m) in enumerate(items, start=1):
                new_suffix = str(order).zfill(width)
                lines[idx] = m.group(1) + m.group(2) + m.group(3) + m.group(4) + new_suffix + m.group(6) + "\n"

    with open(define_path, 'w', encoding='gbk') as f:
        f.writelines(lines)


# Main processing class

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
        """Load data from CSV"""
        encoding = detect_encoding(self.file_path)
        self.raw_df = pd.read_csv(self.file_path, dtype=str, keep_default_na=False, encoding=encoding)

    def process_columns(self) -> None:
        """Process columns"""
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
            except Exception:
                return len(str(s))

        lengths = self.raw_df.apply(lambda col: col.str.strip().apply(calc_byte_len).max()).fillna(0).astype(int)
        self.attribute_df['len'] = lengths.clip(lower=DEFAULT_LEN).values

    def align_data(self) -> None:
        """Align data columns"""
        for col in self.raw_df.columns:
            width = self.attribute_df.loc[self.attribute_df['processed_col'] == col, 'len'].values[0]
            self.raw_df[col] = self.raw_df[col].apply(lambda x: self._gbk_rjust(str(x).strip(), width))

    def _gbk_rjust(self, text: str, width: int) -> str:
        """Right justify text with GBK encoding awareness"""
        current_len = len(text.encode('gbk', errors='ignore'))
        if current_len >= width:
            return text
        return ' ' * (width - current_len) + text

    def analyze_multi_choice(self) -> None:
        """Analyze multiple choice questions"""
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
        """Generate define.stp file content"""
        lines = []
        current_pos = 1
        for _, row in self.attribute_df.iterrows():
            end_pos = current_pos + row['len'] - 1
            prefix = COLUMN_TYPES[row['type']]
            lines.append(f"{prefix} ${row['processed_col']}=${current_pos}-{end_pos},")
            current_pos = end_pos + 2
        return lines

    def generate_make(self) -> List[str]:
        """Generate make.stp file content"""
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
        """Save output files"""
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