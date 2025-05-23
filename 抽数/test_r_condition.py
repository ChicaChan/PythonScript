import re
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LS_v3 import convert_r_condition

# 测试用例，特别关注包含%in%操作符的复合条件
test_cases = [
    # 基本测试
    'data$x',
    'total',
    'normal_condition',
    
    # 简单条件
    'data$x[(data$S1==1)]',
    'data$x[(data$CTCT=="T4")]',
    
    # 包含|操作符的条件
    'data$x[((data$Z4==1 | data$Z4==2 | data$Z4==3))]',
    'data$x[(data$C4C04==1)|(data$C4C04==2)|(data$C4C04==3)]',
    
    # 包含&操作符的条件
    'data$x[(data$AA21==1) & (data$XIAOFEI==1)]',
    
    # 包含%in%操作符的简单条件
    'data$x[(data$A607 %in% c(1,2,3,4,5))]',
    'data$x[(data$CITY %in% c("北京","上海","广州"))]',
    
    # 包含%in%操作符的复合条件 - 重点测试
    'data$x[(data$Z1R5!=0 & data$A607 %in% c(1,2,3,4,5))]',
    'data$x[(data$Z1R5==1 & data$A607 %in% c(1,2,3,4,5) & data$AGE>30)]',
    'data$x[(data$Z1R5==1 | data$A607 %in% c(1,2,3,4,5))]',
    'data$x[((data$Z1R5==1 | data$Z1R5==2) & data$A607 %in% c(1,2,3,4,5))]',
    'data$x[(data$CITY %in% c("北京","上海") & data$AGE>30)]',
    
    # 多个%in%操作符
    'data$x[(data$A607 %in% c(1,2,3) & data$CITY %in% c("北京","上海"))]'
]

# 测试并输出结果
print('测试R语言条件转换 - 特别关注%in%操作符的处理:')
print('=' * 80)

for i, test in enumerate(test_cases, 1):
    converted = convert_r_condition(test)
    print(f'{i}. 原始条件: {test}')
    print(f'   转换结果: {converted}')
    print('-' * 80)

# 特别测试复合条件中的%in%操作符
print('\n特别测试 - 复合条件中的%in%操作符:')
print('=' * 80)

# 测试案例：Z1R5!=0 & A607 %in% c(1,2,3,4,5)
test_case = 'data$x[(data$Z1R5!=0 & data$A607 %in% c(1,2,3,4,5))]'
converted = convert_r_condition(test_case)
print(f'原始条件: {test_case}')
print(f'转换结果: {converted}')
print(f'预期结果: Z1R5!=0 & A607.isin([1, 2, 3, 4, 5])')
print(f'是否符合预期: {"Z1R5!=0 & A607.isin([1, 2, 3, 4, 5])" in converted}')
print('-' * 80)

print('\n所有条件测试完成!')
print('\n总结: 测试了convert_r_condition函数对复合条件中%in%操作符的处理能力。')