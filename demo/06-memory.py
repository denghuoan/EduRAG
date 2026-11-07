import os
import warnings
from dotenv import load_dotenv
from langchain_core.messages import messages_to_dict,messages_from_dict
from langchain_openai import ChatOpenAI
import os
import json
from langchain_community.chat_message_histories import ChatMessageHistory
warnings.filterwarnings("ignore")
load_dotenv(override=True)

model = ChatOpenAI(
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model=os.getenv("DEEPSEEK_MODEL_NAME"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY")
)

history = ChatMessageHistory()

history.add_user_message('四大名著是什么')
history.add_ai_message('1.红楼梦 2.三国演义 3.水浒传 4.西游记')

print(history.messages)

dicts = messages_to_dict(history.messages)

with open('../history.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(dicts, ensure_ascii=False, indent=2))

# 3.从文件中读取出字典，然后将字典转换成消息
# 3.1 读取文件
messages = json.load(open('../history.json', 'r', encoding='utf-8'))
# 3.2 然后将字典转换成消息
chat_messages = messages_from_dict(messages)
print(f'chat_messages-->{chat_messages}')