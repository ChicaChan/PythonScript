# ... existing code ...

class DataProcessor:
    def __init__(self, file_path: str, output_dir: str):
        # ... existing code ...
        self.multi_question_config = {}  # 新增多选题配置存储

    def parse_make_file(self):
        """解析make文件获取多选题配置"""
        make_path = os.path.join(self.output_dir, "make.stp")
        if not os.path.exists(make_path):
            return

        with open(make_path, 'r', encoding='gbk') as f:
            for line in f:
                if match := re.match(r'^(\w+);(\d+);', line.strip()):
                    base, count = match.groups()
                    self.multi_question_config[base] = int(count)

    def adjust_define_order(self, define_lines: List[str]) -> List[str]:
        """调整多选题定义顺序"""
        pattern = re.compile(r'(\w+?)(\d+)=(\$\d+-\d+),?')

        # 创建待调整的题目池
        question_pool = defaultdict(dict)
        other_lines = []

        for line in define_lines:
            if match := pattern.search(line):
                base = match.group(1)
                num = match.group(2)
                if base in self.multi_question_config:
                    question_pool[base][int(num)] = line
                else:
                    other_lines.append(line)
            else:
                other_lines.append(line)

        # 按配置重新排序
        sorted_lines = []
        for base, count in self.multi_question_config.items():
            if base in question_pool:
                expected_nums = range(1, count + 1)
                for n in expected_nums:
                    if n in question_pool[base]:
                        # 保持原有格式只替换题号
                        old_line = question_pool[base][n]
                        new_line = re.sub(
                            fr'{base}\d+',
                            f'{base}{n}',
                            old_line
                        )
                        sorted_lines.append(new_line)

        return sorted_lines + other_lines

    def save_files(self) -> None:
        """文件保存"""
        # ... existing code ...

        # 保存define文件（新增调整逻辑）
        define_path = os.path.join(self.output_dir, "define.stp")
        define_lines = self.generate_define()
        self.parse_make_file()  # 解析make文件
        sorted_define = self.adjust_define_order(define_lines)  # 调整顺序

        with open(define_path, 'w', encoding='gbk') as f:
            f.write('\n'.join(sorted_define))

# ... existing code ...
