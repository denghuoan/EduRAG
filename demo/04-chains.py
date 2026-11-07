import os
import warnings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")
load_dotenv(override=True)
# base_url=os.getenv("DEEPSEEK_BASE_URL")

first_prompt = PromptTemplate(template='我的邻居姓{lastname},她生了一个儿子,给他儿子起个名字', input_variables=['lastname'])
second_prompt = PromptTemplate(template='邻居儿子叫{chidname},给他起个小名', input_variables=['chidname'])

model = ChatOpenAI(
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model=os.getenv("DEEPSEEK_MODEL_NAME"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    max_tokens=1000, # 防止超长回复
    temperature=0 # 0-1
)

first_chain = first_prompt | model
second_chain = second_prompt | model

overall_chain = first_chain | second_chain


print(overall_chain) # 打印整体链对象结构,查看链路组合信息

result = overall_chain.invoke("杨")
print(result.content)
