import pandas as pd
import re
import os

def convert_r_condition(condition):
    """
    将R语言格式的条件转换为pandas可识别的格式
    """
    if condition == 'total':
        return condition
    
    if not condition.startswith('data$'):
        return condition
    
    # 检查是否是R格式的条件
    if condition.startswith('data$x'):
        if condition == 'data$x':
            return 'total'       
        # 提取括号内的条件
        match = re.search(r'\[(.*?)\]', condition)
        if match:
            r_cond = match.group(1)
            
            # 处理%in%操作符
            in_match = re.search(r'(\w+)\s*%in%\s*c\((.*?)\)', r_cond)
            if in_match:
                col = in_match.group(1)
                values = in_match.group(2)
                value_list = [v.strip() for v in values.split(',')]
                for i, v in enumerate(value_list):
                    try:
                        value_list[i] = int(v)
                    except ValueError:
                        try:
                            value_list[i] = float(v)
                        except ValueError:
                            value_list[i] = f"'{v}'"
                return f"{col}.isin({value_list})"
            
            # 替换所有的data$为空
            r_cond = r_cond.replace('data$', '')
            
            # 特殊处理形如 (C104==1)|(C104==2) 的模式
            # 这种模式通常出现在多个OR条件的情况下
            if ')|(' in r_cond:
                # 将 )|( 替换为 ) | (
                r_cond = r_cond.replace(')|(', ') | (')
            
            # 处理括号
            # 首先检查是否有外层括号
            if r_cond.startswith('(') and r_cond.endswith(')'):
                # 检查是否有嵌套括号
                if r_cond.startswith('((') and r_cond.endswith('))'):
                    # 保留一层括号，去掉外层括号
                    r_cond = r_cond[1:-1]
                # 否则保持原样，不去掉括号
            
            # 处理OR条件 (|)
            # 确保|周围有空格
            r_cond = r_cond.replace('|', ' | ')
            r_cond = re.sub(r'\s+\|\s+', ' | ', r_cond)
            
            # 处理AND条件 (&)
            # 确保&周围有空格
            r_cond = r_cond.replace('&', ' & ')
            r_cond = re.sub(r'\s+&\s+', ' & ', r_cond)
            
            return r_cond
    
    return condition

# 用户提供的所有R语言条件
test_cases = [
    'data$x',
    'data$x[(data$S1==1)]',
    'data$x[(data$S1==2)]',
    'data$x[(data$S2==2)]',
    'data$x[(data$S2==3)]',
    'data$x[(data$S2==4)]',
    'data$x[(data$S2==5)]',
    'data$x[(data$S2==6)]',
    'data$x[(data$CTCT=="T4")]',
    'data$x[(data$CTCT=="A1")]',
    'data$x[(data$CTCT=="A")]',
    'data$x[(data$CTCT=="B")]',
    'data$x[(data$CTCT=="C")]',
    'data$x[(data$CTCT=="D")]',
    'data$x[(data$PF=="gdt")]',
    'data$x[(data$PF=="tt")]',
    'data$x[(data$PF=="tuia")]',
    'data$x[((data$Z4==1 | data$Z4==2 | data$Z4==3))]',
    'data$x[((data$Z4==4 | data$Z4==5 | data$Z4==6 | data$Z4==7))]',
    'data$x[((data$Z4==8 | data$Z4==9 | data$Z4==10))]',
    'data$x[(data$AA11==1)]',
    'data$x[(data$AA12==1)]',
    'data$x[(data$AA13==1)]',
    'data$x[(data$AA14==1)]',
    'data$x[(data$AA15==1)]',
    'data$x[(data$AA16==1)]',
    'data$x[(data$AA21==1)]',
    'data$x[((data$B1==9 | data$B1==10 | data$B1==11))]',
    'data$x[((data$B1==1 | data$B1==2 | data$B1==3|data$B1==4 | data$B1==5 | data$B1==6))]',
    'data$x[(data$AA21==1) & (data$XIAOFEI==1)]',
    'data$x[((data$B1==9 | data$B1==10 | data$B1==11))&(data$XIAOFEI==1)]',
    'data$x[((data$B1==1 | data$B1==2 | data$B1==3|data$B1==4 | data$B1==5 | data$B1==6))&(data$XIAOFEI==1)]',
    'data$x[(data$AA22==1)]',
    'data$x[((data$B2==9 | data$B2==10 | data$B2==11))]',
    'data$x[((data$B2==1 | data$B2==2 | data$B2==3|data$B2==4 | data$B2==5 | data$B2==6))]',
    'data$x[(data$AA24==1)]',
    'data$x[((data$B4==9 | data$B4==10 | data$B4==11))]',
    'data$x[((data$B4==1 | data$B4==2 | data$B4==3|data$B4==4 | data$B4==5 | data$B4==6))]',
    'data$x[(data$AA26==1)]',
    'data$x[((data$B6==9 | data$B6==10 | data$B6==11))]',
    'data$x[((data$B6==1 | data$B6==2 | data$B6==3|data$B6==4 | data$B6==5 | data$B6==6))]',
    'data$x[(data$C4C04==1)|(data$C4C04==2)|(data$C4C04==3)|(data$C4C04==4)|(data$C4C04==5)]',
    'data$x[(data$C4C04==4)|(data$C4C04==5)]',
    'data$x[(data$C4C01==1)|(data$C4C01==2)|(data$C4C01==3)|(data$C4C01==4)|(data$C4C01==5)]',
    'data$x[(data$C4C01==4)|(data$C4C01==5)]',
    'data$x[(data$C4C03==1)|(data$C4C03==2)|(data$C4C03==3)|(data$C4C03==4)|(data$C4C03==5)]',
    'data$x[(data$C4C03==4)|(data$C4C03==5)]',
    'data$x[(data$C4E04==1)|(data$C4E04==2)|(data$C4E04==3)|(data$C4E04==4)|(data$C4E04==5)]',
    'data$x[(data$C4E04==4)|(data$C4E04==5)]',
    'data$x[(data$C4E01==1)|(data$C4E01==2)|(data$C4E01==3)|(data$C4E01==4)|(data$C4E01==5)]',
    'data$x[(data$C4E01==4)|(data$C4E01==5)]',
    'data$x[(data$C4E03==1)|(data$C4E03==2)|(data$C4E03==3)|(data$C4E03==4)|(data$C4E03==5)]',
    'data$x[(data$C4E03==4)|(data$C4E03==5)]',
    'data$x[(data$C4F04==1)|(data$C4F04==2)|(data$C4F04==3)|(data$C4F04==4)|(data$C4F04==5)]',
    'data$x[(data$C4F04==4)|(data$C4F04==5)]',
    'data$x[(data$C4F01==1)|(data$C4F01==2)|(data$C4F01==3)|(data$C4F01==4)|(data$C4F01==5)]',
    'data$x[(data$C4F01==4)|(data$C4F01==5)]',
    'data$x[(data$C4F03==1)|(data$C4F03==2)|(data$C4F03==3)|(data$C4F03==4)|(data$C4F03==5)]',
    'data$x[(data$C4F03==4)|(data$C4F03==5)]',
    'data$x[(data$AA23==1)]',
    'data$x[((data$B3==9 | data$B3==10 | data$B3==11))]',
    'data$x[((data$B3==1 | data$B3==2 | data$B3==3|data$B3==4 | data$B3==5 | data$B3==6))]',
    'data$x[(data$C104==1)|(data$C104==2)|(data$C104==3)|(data$C104==4)|(data$C104==5)]',
    'data$x[(data$C104==4)|(data$C104==5)]'
]

# 测试并输出结果
print('测试R语言条件转换:')
print('=' * 80)
for i, test in enumerate(test_cases, 1):
    converted = convert_r_condition(test)
    print(f'{i}. 原始条件: {test}')
    print(f'   转换结果: {converted}')
    print('-' * 80)

print('\n所有条件测试完成!')