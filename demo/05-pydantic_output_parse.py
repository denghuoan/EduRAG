# 定义Pydantic模型
import os
import warnings
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from langchain_openai import ChatOpenAI
# pydantic
from pydantic import BaseModel, Field
from typing import List


warnings.filterwarnings("ignore")
load_dotenv(override=True)

model = ChatOpenAI(
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model=os.getenv("DEEPSEEK_MODEL_NAME"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY")
)

class Movie(BaseModel):
    title: str = Field(description="电影标题")
    director: str = Field(description="导演姓名")
    year: int = Field(description="上映年份")
    # todo : 单标签多分类 （每个电影只能有个类型）、 多标签多分类（一个电影可以有多个类型）
    genre: List[str] = Field(description="电影类型")
    rating: float = Field(description="评分（1-10）")

# 创建Pydantic解析器
pydantic_parser = PydanticOutputParser(pydantic_object=Movie)

# 创建带格式说明的提示模板
format_instructions = pydantic_parser.get_format_instructions()
print(format_instructions)

pydantic_prompt = ChatPromptTemplate.from_template(
    "生成一部{genre}电影的信息。\n{format_instructions}"
)

# 组合组件
pydantic_chain = pydantic_prompt | model | pydantic_parser

# 调用链
movie_data = pydantic_chain.invoke({
    "genre": "科幻",
    "format_instructions": format_instructions
})
print(movie_data)