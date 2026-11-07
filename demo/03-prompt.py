import os
import warnings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
warnings.filterwarnings("ignore")
load_dotenv(override=True)
# base_url=os.getenv("DEEPSEEK_BASE_URL")

model = ChatOpenAI(
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model=os.getenv("DEEPSEEK_MODEL_NAME"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    max_tokens=1000,
    temperature=0
)

template = "我的邻居姓{lastname},她生了个儿子,给她儿子起个名字"
prompt = PromptTemplate(template=template, input_variables=['lastname'])

prompt_text = prompt.format(lastname='杨')
res = model.invoke(prompt_text)
print(res)

# result = model.invoke('四大名著是哪四本书')
# print(result.content)

