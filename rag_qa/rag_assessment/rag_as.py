# 该脚本用于: 通过RAGAS框架 对RAG系统做评分, 主要考察: 检索部分 和 生成部分.
# 一共有4个指标, 分别是:
#   生成部分: 忠实度(忠诚度), 答案相关性.
#   检索部分: 上下文召回率(召回), 上下文精确率(准确).

# todo 1.导包
from ragas import evaluate          # 导入ragas库的evaluate函数，用于执行RAG评估
from ragas.metrics import (         # 导入ragas的评估指标，包括忠实度、答案相关性、上下文相关性和上下文召回率
    faithfulness,           # 忠实度
    answer_relevancy,       # 答案相关性
    context_precision,      # 上下文精确度
    context_recall          # 上下文召回率
)
from datasets import Dataset        # 导入datasets库的Dataset类，用于构建RAGAS所需的数据格式
import json                 # 导入json库，用于加载JSON格式的评估数据集

# 导入langchain_openai的聊天模型 和 嵌入模型，用于评估时的语义计算和推理
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 随堂我用本地的ollama模型.
from langchain_ollama import ChatOllama, OllamaEmbeddings

# todo 1. 加载评估数据集
with open('rag_evaluate_data.json', 'r', encoding='utf8') as f :
    data = json.load(f)

# todo 2. 转换数据格式 Dataset
eval_data = {
    'question': [item['question'] for item in data],
    'answer': [item['answer'] for item in data],
    'contexts': [item['context'] for item in data],
    'ground_truth': [item['ground_truth'] for item in data],
}
dataset = Dataset.from_dict(eval_data)

# todo 3. 配置评估模型
llm = ChatOllama(model='qwen3.5:4b', base_url='http://localhost:11434')
embeddings = OllamaEmbeddings(model='qwen3.5:4b', base_url='http://localhost:11434')

# todo 4. 执行评估
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=llm,
    embeddings=embeddings
)

print(f'RAGAS评估结果: {result}')






