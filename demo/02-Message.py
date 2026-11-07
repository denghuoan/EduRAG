import os
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
# 忽略警告
import warnings
warnings.filterwarnings('ignore')

load_dotenv(override=True)

model = init_chat_model(
    os.getenv("DEEPSEEK_MODEL_NAME"),
    model_provider="openai",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0,  # 0-1
)
messages = [
    SystemMessage(content='你是一个学识渊博的文学大师'),
    HumanMessage(content='四大名著是哪四本书'),
    AIMessage(content='1. 《红楼梦》2. 《西游记》3. 《水浒传》4. 《三国演义》'),
    HumanMessage(content='四大发明是什么')]

# messages = [
#     {"role": "system", "content": "你是一个学识渊博的文学大师"},
#     {"role": "user", "content": "四大名著是哪四本书"},
#     {"role": "assistant", "content": "1. 《红楼梦》2. 《西游记》3. 《水浒传》4. 《三国演义》"},
#     {"role": "user", "content": "四大发明是什么"},
# ]
result = model.invoke(messages)

print(result.content)