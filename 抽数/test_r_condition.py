import pandas as pd
import re

def convert_r_condition(condition):
    """将R语言格式的条件转换为Python pandas可识别的格式"""
    # 如果条件已经是'total'，直接返回
    if condition == 'total':
        return condition
    
    # 如果条件已经是pandas格式（不是R格式），直接返回
    if not condition.startswith('data$'):
        return condition
    
    # 检查是否是R格式的条件
    if condition.startswith('data$x'):
        # 处理data$x，表示total
        if condition == 'data$x':
            return 'total'
        
        # 提取括号内的条件
        match = re.search(r'\[(.*?)\]', condition)
        if match:
            r_cond = match.group(1)
            
            # 替换data$为空
            r_cond = r_cond.replace('data$', '')
            
            # 处理%in%操作符，例如：J3 %in% c(6,7) -> J3.isin([6, 7])
            in_match = re.search(r'(\w+)\s*%in%\s*c\((.*?)\)', r_cond)
            if in_match:
                col = in_match.group(1)
                values = in_match.group(2)
                # 将逗号分隔的值转换为列表
                value_list = [v.strip() for v in values.split(',')]
                # 尝试将数值转换为整数或浮点数
                for i, v in enumerate(value_list):
                    try:
                        value_list[i] = int(v)
                    except ValueError:
                        try:
                            value_list[i] = float(v)
                        except ValueError:
                            # 如果不是数值，保持字符串格式
                            value_list[i] = f"'{v}'"
                return f"{col}.isin({value_list})"
            
            # 处理括号内的条件，例如：(data$S1==1) -> S1==1
            # 或者 ((data$S2a==15 | data$S2a==16)) -> (S2a==15 | S2a==16)
            if r_cond.startswith('(') and r_cond.endswith(')'):
                # 检查是否有嵌套括号
                if r_cond.startswith('((') and r_cond.endswith('))'):
                    # 保留一层括号，去掉外层括号
                    r_cond = r_cond[1:-1]
                else:
                    # 去掉所有括号
                    r_cond = r_cond[1:-1]
            
            # 处理多个OR条件，例如：(S2a==15 | S2a==16 | ...) -> (S2a==15 | S2a==16 | ...)
            # 在pandas中，|操作符可以直接用于布尔表达式，但需要确保有空格
            r_cond = r_cond.replace('|', ' | ')
            # 修复可能出现的多余空格
            r_cond = re.sub(r'\s+\|\s+', ' | ', r_cond)
            
            # 处理多个AND条件，例如：(PIN01==1&QA6_1==1) -> (PIN01==1 & QA6_1==1)
            # 将&替换为空格&空格，以符合pandas语法
            r_cond = r_cond.replace('&', ' & ')
            r_cond = re.sub(r'\s+&\s+', ' & ', r_cond)
            
            return r_cond
    
    # 如果不是R格式，或者无法解析，返回原始条件
    return condition

# 测试用例 - 包含用户提供的所有R语言条件示例
test_cases = [
    'data$x',
    'data$x[(data$S1==1)]',
    'data$x[(data$S1==2)]',
    "data$x[(data$ctct=='T4')]",
    "data$x[(data$ctct=='A1')]",
    "data$x[(data$ctct=='A')]",
    "data$x[(data$ctct=='B')]",
    "data$x[(data$ctct=='C')]",
    "data$x[(data$ctct=='D')]",
    'data$x[((data$S2a==15 | data$S2a==16 | data$S2a==17 | data$S2a==18 | data$S2a==19))]',
    'data$x[((data$S2a==20 | data$S2a==21 | data$S2a==22 | data$S2a==23 | data$S2a==24 | data$S2a==25 | data$S2a==26 | data$S2a==27 | data$S2a==28 | data$S2a==29))]',
    'data$x[((data$S2a==30 | data$S2a==31 | data$S2a==32 | data$S2a==33 | data$S2a==34 | data$S2a==35 | data$S2a==36 | data$S2a==37 | data$S2a==38 | data$S2a==39))]',
    'data$x[((data$S2a==40 | data$S2a==41 | data$S2a==42 | data$S2a==43 | data$S2a==44 | data$S2a==45 | data$S2a==46 | data$S2a==47 | data$S2a==48 | data$S2a==49))]',
    'data$x[((data$S2a==50 | data$S2a==51 | data$S2a==52 | data$S2a==53 | data$S2a==54 | data$S2a==55 | data$S2a==56 | data$S2a==57 | data$S2a==58 | data$S2a==59))]',
    'data$x[((data$S2a==60 | data$S2a==61 | data$S2a==62 | data$S2a==63 | data$S2a==64 | data$S2a==65))]',
    "data$x[(data$pf=='tt')]",
    "data$x[(data$pf=='gdt')]",
    "data$x[(data$pf=='tuia')]",
    'data$x[data$J3 %in% c(6,7)]',
    'data$x[(data$PIN01==1)]',
    'data$x[(data$A3_1==1)]',
    'data$x[(data$A3_2==1)]',
    'data$x[(data$A3_4==1)]',
    'data$x[(data$A3_10==1)]',
    'data$x[(data$A3_4==1 | data$A3_8==1)]',
    'data$x[(data$A4_1==1)]',
    'data$x[(data$A4_2==1)]',
    'data$x[(data$A3_7==1)]',
    'data$x[(data$A4_11==1)]',
    'data$x[(data$PIN01==1&data$QA6_1==1)]',
    'data$x[(data$PIN02==1&(data$A3A4048==2))]',
    'data$x[(data$PIN01==1&data$QZ2R1==2)]',
    'data$x[(data$PIN07==1&(data$A3A4_10==1))]',
    "data$x[(data$pf=='gdt'&data$PIN02==1)]",
    'S1==1',  # 已经是pandas格式
    "ctct=='T4'"  # 已经是pandas格式
]

print('测试R语言条件转换:')
for test in test_cases:
    converted = convert_r_condition(test)
    print(f'原始条件: {test}')
    print(f'转换结果: {converted}')
    print('-' * 50)