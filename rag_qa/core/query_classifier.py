# 该脚本用于: 基于BERT模型实现查询分类器, 将用户查询分为'通用知识'和'专业咨询'两大类.
# 核心流程: 加载预训练BERT模型 -> 数据预处理 -> 模型训练 -> 模型评估 -> 预测查询类别
# 应用场景: 在RAG系统中, 可根据分类结果决定是否需要检索专业文档(例如: 专业咨询可检索文档, 通用知识由模型直接回复即可)


# todo 1.导包
# 导入标准库
import json
import os, sys

# 路径配置: 解决跨目录模块导入问题 -> 定位项目各层级目录.
current_dir= os.path.dirname(os.path.abspath(__file__))   # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa\core
rag_qa_path = os.path.dirname(current_dir)      # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa
project_root = os.path.dirname(rag_qa_path)     # D:\workspace\ai_30_bj\integrated_qa_system
# 将项目目录添加到系统路径中
sys.path.insert(0, project_root)

# 导入 PyTorch
import torch
# 导入日志
from base.logger import logger
# 导入numpy
import numpy as np
# 导入 Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification       # BERT分词器和序列分类模型.
# Trainer: 负责实际执行训练过程(用模型, 数据和 TrainingArguments的配置跑训练) -> 相当于 训练任务的执行者, 按照 TrainingArguments(说明书)具体干活
# TrainingArguments: 负责定义训练的各种参数和配置(例如: 训练多久, 多大批次, 保存到哪等...) -> 相当于 训练任务的说明书,提前写好训练的各种规则和细节.
from transformers import Trainer, TrainingArguments
# 导入train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# todo 2.定义QueryClassifier类: 封装BERT查询分类的完整流程 -> 模型加载, 训练, 评估, 预测...
class QueryClassifier:
    # todo 2.1 初始化方法: 配置模型路径, 加载分词器, 选择设备, 定义标签映射.
    def __init__(self, model_path='../models/bert_query_classifier'):
        # 1. 存储模型路径: 用于后续加载或保存模型.
        self.model_path = model_path
        # 2. 加载BERT分词器: 将文本转换成模型可以理解的输入.
        # 2.1 拼接预训练BERT模型的本地路径.
        bert_path = os.path.join(rag_qa_path, 'models', 'bert-base-chinese')
        # 2.2 加载分词器.
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        # 3. 初始化模型变量, 后续通过 load_model()加载或创建模型.
        self.model = None

        # 4. 选择模型运行设备.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 5. 记录日志.
        logger.info(f'使用设备: {self.device}')

        # 6. 定义标签映射: 将标签映射为模型输出的索引. 即: 通用知识 -> 0, 专业咨询 -> 1.
        self.label_map = {"通用知识": 0, "专业咨询": 1}

        # 7. 加载模型: 初始化时, 自动调用load_model(), 确保模型可用.
        self.load_model()


    # todo 2.2 加载模型方法 -> 从指定路径加载已训练模型, 若不存在则初始化新模型.
    def load_model(self):
        """
        函数功能: 加载模型, 优先从self.model_path加载, 不存在则初始化基于 bert-base-chinese的新模型
        :return:
        """
        # 1. 检查模型路径是否存在 -> 即: 是否有已训练的模型.
        if os.path.exists(self.model_path):
            # 加载预训练模型
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录加载成功的日志
            logger.info(f"加载模型: {self.model_path}")
        else:
            # 2. 若模型不存在, 初始化新模型
            # 参1: 模型路径,  参2: 二分类任务.
            self.model = BertForSequenceClassification.from_pretrained("../models/bert-base-chinese", num_labels=2)
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录初始化模型的日志
            logger.info("初始化新 BERT 模型")


    # todo 2.9 预测类别方法 -> 对单个查询文本进行分类预测.
    def predict_category(self, query):
        """
        函数功能: 对单个查询文本进行类别预测 -> 通用知识 或者 专业咨询.
        :param query: 待分类的查询文本
        :return: 类别名称(文本标签)
        """
        # 1. 检查模型是否加载.
        if self.model is None:
            # 模型未加载
            logger.error("模型未训练或加载!")
            return '通用知识'

        # 2. 对查询文本进行编码.
        encoding = self.tokenizer(
            query,
            truncation=True, # 超过最大长度时截断
            padding=True, # 不足最大长度时填充
            max_length=128,
            return_tensors="pt" # 返回张量
        )

        # 3. 将编码数据移到指定设备.
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        print(f'encoding: {encoding}')
        # 4. 模型推理 -> 不计算梯度, 提高效率.
        with torch.no_grad():
            # 4.1 获取模型输出 -> 包含logits
            outputs = self.model(**encoding)
            # 4.2 获取预测结果 -> 获取logits最大值对应的索引作为预测结果.
            prediction = torch.argmax(outputs.logits, dim=1).item()

        # 5. 返回预测结果标签
        return '专业咨询' if prediction == 1 else '通用知识'

# todo 3. 测试代码.
if __name__ == '__main__':
    # 1. 实例化查询分类器 -> 自动加载模型, 若无则初始化新模型.
    query_classify = QueryClassifier()

    # 3. 示例预测: 对查询'AI的课程大纲是什么'进行分类
    # result = query_classify.predict_category('大数据分析师需要考取哪些证书')
    result = query_classify.predict_category('AI的课程大纲是什么')
    print(result)