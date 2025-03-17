import pandas as pd
import numpy as np
from pulp import *
import re
import os


class SampleSelector:
    def __init__(self, input_path: str):
        """初始化采样器"""
        self.input_path = input_path
        self.all_data = None
        self.raw_data = None
        self.quota_limit = None
        self.result = None

    def load_data(self):
        """加载输入数据"""
        # 读取Excel的两个sheet
        self.all_data = pd.read_excel(self.input_path, sheet_name=0)
        self.raw_data = self.all_data.iloc[:, 1:]
        self.quota_limit = pd.read_excel(self.input_path, sheet_name=1)

    def _convert_r_constraint(self, r_expr: str) -> str:
        """转换R语言约束表达式为Python表达式"""
        # 如果表达式仅为 data$x，返回整个数据框的布尔条件
        if r_expr == 'data$x':
            return 'pd.Series([True] * len(data))'

        # 处理包含多个OR条件的表达式
        if '|' in r_expr:
            # 检查是否包含S2a的范围条件
            match = re.search(r'data\$x\[\(\(data\$(\w+)==([\d\s\|=]+)\)\)\]', r_expr)
            if match:
                col_name = match.group(1)
                # 提取所有数值
                values = [int(num) for num in re.findall(r'==(\d+)', r_expr)]
                # 返回isin条件
                return f"data['{col_name}'].isin({values})"

        # 处理简单的等于条件
        match = re.search(r'data\$x\[\(data\$(\w+)==[\'\"]?([^\'\"\)]+)[\'\"]?\)\]', r_expr)
        if match:
            col_name, value = match.groups()
            # 如果值是数字，直接比较；否则加引号
            if value.isdigit():
                return f"data['{col_name}'] == {value}"
            else:
                return f"data['{col_name}'] == '{value}'"

        # 默认返回全True条件
        return 'pd.Series([True] * len(data))'

    def solve_lp(self):
        """求解线性规划问题"""
        # 构建基础数据框
        if self.raw_data.shape[1] > 1:
            group_cols = self.raw_data.columns.tolist()
            data = self.raw_data.groupby(group_cols).size().reset_index(name='raw')
        else:
            data = pd.DataFrame(self.raw_data.iloc[:, 0].value_counts()).reset_index()
            data.columns = [self.all_data.columns[1], 'raw']

        # 创建线性规划问题
        prob = LpProblem("Sample_Selection", LpMaximize)

        # 创建决策变量
        x = LpVariable.dicts("x", range(len(data)), lowBound=0, upBound=None, cat='Integer')

        # 目标函数：最大化样本数
        prob += lpSum(x.values())

        # 添加约束条件
        for _, row in self.quota_limit.iterrows():
            target = row.iloc[1]  # target value
            max_val = row.iloc[2]  # upper bound
            min_val = row.iloc[3]  # lower bound
            constraint_expr = row.iloc[0]  # R expression

            try:
                # 转换R约束为Python约束
                py_expr = self._convert_r_constraint(constraint_expr)

                # 对于简单条件，直接评估
                eval_globals = {'data': data, 'pd': pd}
                mask = eval(py_expr, eval_globals)

                # 获取满足条件的索引
                constrained_indices = data[mask].index

                # 添加约束
                if min_val > 0:
                    prob += lpSum(x[i] for i in constrained_indices) >= min_val
                if not pd.isna(max_val):
                    prob += lpSum(x[i] for i in constrained_indices) <= max_val

            except Exception as e:
                print(f"Warning: Skipping constraint '{constraint_expr}' due to error: {e}")
                continue

        # 求解
        status = prob.solve()

        # 处理结果
        data['solution'] = [int(value(x[i])) for i in range(len(data))]
        self.result = data

    def save_results(self):
        """保存结果到Excel"""
        # 准备输出数据
        samples_select = self.all_data.copy()
        samples_select['select'] = 0

        # 根据结果标记选中的样本
        for idx, row in self.result.iterrows():
            mask = (self.raw_data == row[self.raw_data.columns]).all(axis=1)
            samples_select.loc[mask, 'select'] = row['solution'] if row['solution'] > 0 else 0

        # 保存到Excel，并计算每个约束的实际值
        with pd.ExcelWriter('output.xlsx') as writer:
            samples_select.to_excel(writer, sheet_name='samples_select', index=False)
            self.result.to_excel(writer, sheet_name='result', index=False)

            # 计算每个约束的实际值
            actuals = []
            for _, row in self.quota_limit.iterrows():
                constraint_expr = row.iloc[0]
                py_expr = self._convert_r_constraint(constraint_expr)

                try:
                    if constraint_expr == 'data$x':
                        # 总样本数
                        actual = self.result['solution'].sum()
                    else:
                        # 计算满足约束条件的样本数
                        eval_globals = {'data': self.result, 'pd': pd}
                        mask = eval(py_expr, eval_globals)
                        actual = self.result.loc[mask, 'solution'].sum()
                except Exception as e:
                    print(f"Warning: Error calculating actual for '{constraint_expr}': {e}")
                    actual = 0

                actuals.append(actual)

            # 创建比较表
            compare = pd.DataFrame({
                'constraint': self.quota_limit.iloc[:, 0],
                'target': self.quota_limit.iloc[:, 1],
                'actual': actuals
            })
            compare.to_excel(writer, sheet_name='compare', index=False)

    def run(self):
        """运行完整流程"""
        self.load_data()
        self.solve_lp()
        self.save_results()

        if self.result['solution'].sum() > 0:
            print("Success!")
        else:
            print("Fail~")


if __name__ == "__main__":
    selector = SampleSelector('input.xlsx')
    selector.run()
