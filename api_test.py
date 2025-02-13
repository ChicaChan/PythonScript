# -*- coding: utf-8 -*-
from openai import OpenAI
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gbk')

client = OpenAI(api_key="sk-671f95339e314ff2bf358f6e6036dbff", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": "You are a helpful assistant and please answer me in Chinese"},
        {"role": "user", "content": "介绍一下品牌资产指数(BEI),并介绍计算方式"},
    ],
    stream=False
)

print(response.choices[0].message.content)