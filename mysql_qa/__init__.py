import os, sys
current_dir = os.path.abspath(__file__)
# print(current_dir)
mysql_qa_path = os.path.dirname(current_dir)
# print(rag_qa_path)

# 添加rag_qa_path路径到系统路径变量中
sys.path.insert(0, mysql_qa_path)

# 具体的导包动作port RAGSystem
from db.mysql_client import MySQLClient
from cache.redis_client import RedisClient
from retrieval.bm25_search import BM25Search