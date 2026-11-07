# 核心流程:
# 1.文档向量生成(BGE-M3)
# 2.向量入库
# 3.混合检索(稠密+稀疏向量)
# 4.结果去重
# 5.重排序
# 6.返回精准文档
import hashlib

import torch.cuda

# 常用的检索策略介绍
# 1.密集检索: 语义级匹配
# 2.稀疏检索: 关键词过滤
# 3.多向量检索: 逐词细节对比

import os,sys

from langchain_community.vectorstores import Milvus
from milvus_model.hybrid import BGEM3EmbeddingFunction
from openai.types.beta.chatkit.chatkit_response_output_text import AnnotationFile
from sentence_transformers import CrossEncoder
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker

# todo 1. 路径配置模块. 定位当前脚本, RAG核心目录, 项目根目录, 确保自定义模块可导入.
# 1.获取当前文件所在目录的绝对路径.
local_path = os.path.abspath(os.path.dirname(__file__))
# print(f'local_path: {local_path}')      # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa\core

# 2. 获取当前目录的上级目录.
rag_qa_path = os.path.abspath(os.path.dirname(local_path))
# print(f'module_dir: {rag_qa_path}')     # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa

# 3. 获取项目根目录的绝对路径.
project_root = os.path.dirname(rag_qa_path)
# print(f'project_root: {project_root}')  # D:\workspace\ai_30_bj\integrated_qa_system

# 4. 将项目根目录加入系统路径, 确保跨目录导入自定义模块(例如: config, logger)...
sys.path.insert(0, project_root)

from base.config import Config
from base.logger import logger
from rag_qa.core.document_processor import process_documents
from langchain_core.documents import Document
conf = Config()
class VectorStore:
    def __init__(
            self,
            collection_name=conf.MILVUS_COLLECTION_NAME,
            host=conf.MILVUS_HOST,
            port=conf.MILVUS_PORT,
            database=conf.MILVUS_DATABASE_NAME
    ):
        # todo 1.Milvus核心设置
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.database = database

        self.logger = logger
        # 选择模型运行设备 -> 优先GPU,无GPU则用CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'使用设备: {self.device}')

        # todo 2.初始化BGE-Reranker重排序模型 --> 用于计算'查询-文档'的相关性得分
        reranker_path = os.path.join(rag_qa_path, 'models', 'bge-reranker-large')
        self.reranker = CrossEncoder(reranker_path, device=self.device)

        # todo 3.初始化BGE-M3模型 --> 用于生成文档和查询的向量表示
        m3_path = os.path.join(rag_qa_path, 'models', 'bge-m3')
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=m3_path,
            use_fp16=(self.device == 'cuda'), # GPU时启用半精度计算(减少内存占用, 提升速度), CPU时禁用
            device=self.device
        )

        # todo 4.获取BGE-M3稠密向量的纬度: 固定输出1024纬稠密向量
        self.dense_dim = self.embedding_function.dim['dense']
        # print(f'稠密向量的维度: {self.dense_dim}')  # 结果: 1024

        # todo 5.初始化Milvus客户端
        self.client = MilvusClient(
            uri=f'http://{self.host}:{self.port}',
            db_name=self.database
        )

        # todo 6.加载集合,无则创建
        self._create_or_load_collection()


    def _create_or_load_collection(self):
        if not self.client.has_collection(self.collection_name):

            # todo 1.创建Schema
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # todo 2.创建索引参数 稠密和稀疏两路
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加IVF_FLAT索引,度量类型为内积(IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name='dense_index',
                index_type='IVF_FLAT',
                metric_type='IP',
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                index_name="sparse_index",
                field_name="sparse_vector",
                index_type='SPARSE_INVERTED_INDEX',
                metric_type='IP',
                params={"drop_ratio_build": 0.2} # 构建索引时,丢弃20%的低权重值(减少存储,噪声,精度影响较小)
            )

            # todo 3.创建Milvus集合
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )

            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")

        else:
            logger.info(f"已加载集合 {self.collection_name}")

        # 将集合加载到内存,确保可立即查询
        self.client.load_collection(self.collection_name)

    def add_document(self, documents):
        texts = [doc.page_content for doc in documents]
        embeddings= self.embedding_function(texts)

        data=[]

        for i,doc in enumerate(documents):
            # encode('utf-8') 将字符串转成字节流,因为md5哈希需要字节
            # hexdigest() 将哈希转为16进制字符串(32位,适合作为ID)
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()

            # 密集向量dense --> 抓整体语义
            # 稀疏向量sparse --> 抓关键词权重
            # embedding_function.dim --> {'dense':1024, 'colbert_vecs':1024, 'sparse':250002}

            sparse_vector = {}
            # 获取第i个文档的稀疏向量行
            row = embeddings['sparse'][[i]]
            # 获取稀疏向量的非零值索引 和对应的权重
            indics = row.indices
            values = row.data
            # 组装稀疏向量字典,将索引和权重配对
            for idx, value in zip(indics, values):
                sparse_vector[idx] = value


            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": embeddings['dense'][i],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata['parent_id'],
                "parent_content": doc.metadata['parent_content'],
                "source": doc.metadata.get('source', "unknown"),
                "timestamp": doc.metadata.get('timestamp', "unknown"),
            })

        if data:
            self.client.upsert(
                collection_name=self.collection_name,
                data=data
            )
            logger.info(f'已插入/更新 {len(data)} 条数据到集合 {self.collection_name}')


    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        """
        混合检索(稠密+稀疏)+结果重排序,返回精准父文档
        :param query:
        :param k:
        :param source_filter: 学科过滤条件, 例如: ai, 仅检索该学科的文档, 默认None不过滤.
        :return: 重排序后的Top-M个父文档列表(M是从conf中获取的)
        """
        # 生成query的嵌入(需要传列表, 不能直接传字符串)
        query_embeddings = self.embedding_function([query])
        # query的稠密向量
        dense_query_vector = query_embeddings['dense'][0]
        # query的稀疏向量
        sparse_query_vector = {}
        # 获取query稀疏向量的第0行数据 -> 仅1个查询, 故取第0行
        row = query_embeddings['sparse'][[0]]
        indics = row.indices # 获取稀疏向量的非零值索引
        values = row.data # 获取稀疏向量的非零值
        for idx, value in zip(indics, values):
            sparse_query_vector[idx] = value


        # 构建检索过滤表达式:按学科过滤
        filter_expr = f"source == '{source_filter}'" if source_filter else ""


        # 定义稠密向量的检索参数
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field='dense_vector', # 检索的向量字段
            param={
                'metric_type': 'IP',
                'params': {'nprobe': 10}  # 内积相似度,检索时访问的聚类数(平衡速度和精度)
            },
            limit=k, # 返回Tok-K结果
            expr=filter_expr # 应用过滤表达式
        )

        # 稀疏向量的检索参数
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field='sparse_vector',
            param={
                'metric_type': 'IP',
                'params': {} # 稀疏向量无需额外参数
            },
            limit=k,
            expr=filter_expr
        )

        # 创建加权排序器
        ranker = WeightedRanker(1.0, 0.7) # 稠密权重1.0  稀疏权重0.7
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=['text', 'parent_id', 'parent_content', 'source', 'timestamp']
        )[0]

        print(f'results: {results}, 类型: {type(results)}, 数据量: {len(results)}')    # 列表类型.

        # 将检索结果转为 LangChain Document对象
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
        print(f'子块列表: {sub_chunks}, 数据量: {len(sub_chunks)}')      # [Document对象, Document对象, ...]

        parent_docs = self._get_unique_parent_docs(sub_chunks)

        if len(parent_docs) < 2:
            return parent_docs[:conf.RETRIEVAL_K]

        if parent_docs:
            # '查询-文档'配对列表 每个配对为 [query, doc_content]
            pairs = [[query, doc.page_content] for doc in parent_docs]
            scores = self.reranker.predict(pairs)

            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]

        else:
            ranked_parent_docs = []


        return ranked_parent_docs[:conf.CANDIDATE_M]

    def _get_unique_parent_docs(self, sub_chunks):
        parent_contens = set()
        unique_docs = []
        for chunk in sub_chunks:
            parent_content = chunk.metadata.get('parent_content', chunk.page_content) # 默认为子块内容
            if parent_content and parent_content not in parent_contens:
                unique_docs.append(Document(
                    page_content= parent_content,
                    metadata=chunk.metadata
                ))
                parent_contens.add(parent_content)
        return unique_docs

    # 将Milvus结果(hit)转换为LangChain Document对象
    def _doc_from_hit(self, hit):
        return Document(
            page_content=hit.get('text'),
            metadata={
                "parent_id": hit.get('parent_id'),
                "parent_content": hit.get('parent_content'),
                "source": hit.get('source'),
                "timestamp": hit.get('timestamp'),
            }
        )



if __name__ == '__main__':
    vector_store = VectorStore()
    # directory_path = os.path.join(rag_qa_path, 'data', 'ai_data')
    # documents = process_documents(directory_path)
    # vector_store.add_document(documents)

    query = 'AI学科的课程内容是什么'
    results = vector_store.hybrid_search_with_rerank(query, source_filter='ai')
    print(f'results: {results}')  # 结果列表: Document对象
    print(f'results数量: {len(results)}')  # 结果数量: 0 ~ 2区间