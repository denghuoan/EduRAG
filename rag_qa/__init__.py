import os, sys
current_dir = os.path.abspath(__file__)
# print(current_dir) # ...eduRAG/rag_qa/__init__.py
rag_qa_path = os.path.dirname(current_dir)
# print(rag_qa_path) # ...eduRAG/rag_qa

# 添加rag_qa_path路径到系统路径变量中
sys.path.insert(0, rag_qa_path)

# 具体的导包动作
from core.prompts import RAGPrompts
from core.vector_store import VectorStore
# from core.rag_system import RAGSystem
from core.new_rag_system import RAGSystem
