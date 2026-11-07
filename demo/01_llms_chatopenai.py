import os
import warnings
from dotenv import load_dotenv
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI
warnings.filterwarnings("ignore")
load_dotenv(override=True)

def use_chat_openai():
    model = ChatOpenAI(
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        model=os.getenv("DEEPSEEK_MODEL_NAME"),
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        max_tokens=1000, # 防止超长回复
        temperature=0 # 0-1
    )
    result = model.invoke('四大名著是哪四本书')
    print(result.content)

def use_ollama():
    model = Ollama(model='qwen3.5:4b')
    print(model.invoke('四大名著是哪四本书'))

if __name__ == "__main__":
    use_chat_openai()