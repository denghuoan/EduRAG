# 导包
import os       # 文件操作
from langchain_community.document_loaders import TextLoader     # 加载纯文本文件(.txt)用的工具
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader    # 加载(.md)文件的工具, 保留标题,列表等结构化信息
from langchain_text_splitters import MarkdownTextSplitter       # 针对 .md文档的文本分割器.
from datetime import datetime               # 用于记录文档加载/处理时间.
from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter        # 适配中文教育文本层级结构
from rag_qa.edu_text_spliter import AliTextSplitter                     # 阿里中文优化(教育场景专用文本分割器)
from rag_qa.edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader  # 教育场景专用文档加载器(支持OCR, 处理扫描版课件, 图片文字等.)
from base.config import Config      # 配置类
from base.logger import logger      # 日志类

conf = Config()

document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}

# todo 1. 加载文档 -> 从指定目录递归加载所有支持类型文件, 添加原数据并返回文档列表.
def load_documents_from_directory(directory_path):
    """
    从指定目录加载所有支持类型的文件,为每个文件添加元数据
    :param directory_path: 目标文件夹的绝对路径
    :return: 加载完成的文档列表,每个元素为 LangChain Document
    """
    documents = []

    # 获取支持的文件扩展名
    supported_extensions = document_loaders.keys()
    # 从目录名提取学科类别元数据 例如 ai_data -> 提取ai作为source
    source = os.path.basename(directory_path).replace("_data", "")

    # 4. 递归遍历目标目录及其所有子目录,  注意: os.walk()返回 当前目录路径, 子目录列表, 文件列表.
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in supported_extensions:
                try:
                    # 根据扩展名 创建加载器实例
                    loader_class = document_loaders[file_extension]
                    if file_extension == ".txt":
                        loader = loader_class(file_path, encoding='utf-8')
                    else:
                        loader = loader_class(file_path)

                    # 加载文件内容 -> 返回文档列表,单个文件可能拆分成多个Document
                    loaded_docs = loader.load()
                    # 给每个加载的文档添加元数据
                    for doc in loaded_docs:
                        doc.metadata['source'] = source
                        doc.metadata['file_path'] = file_path
                        doc.metadata['timestamp'] = datetime.now().isoformat()

                    documents.extend(loaded_docs)
                    logger.info(f'成功加载文件: {file_path}')
                except Exception as e:
                    logger.error(f'加载文件 {file_path} 失败: {str(e)}')

            # 文件不在支持列表
            else:
                logger.warning(f'不支持的文件类型: {file_path}')

    return documents


# todo 2.处理文档 -> 分层切分(父块+子块): 返回带元数据的子块列表
def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                     child_chunk_size=conf.CHILD_CHUNK_SIZE,
                     chunk_overlap=conf.CHUNK_OVERLAP):
    """
    :param directory_path:
    :param parent_chunk_size:
    :param child_chunk_size:
    :param chunk_overlap:
    :return: child_chunks 切分后的子块列表(含父块关联信息)
    """

    documents = load_documents_from_directory(directory_path)
    logger.info(f"加载的文档数量: {len(documents)}")

    # 初始化父块和子块分词器
    # --通用
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # --markdown
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    child_chunks = []

    for i, doc in enumerate(documents):
        # 判断是否是md文档
        file_extension = os.path.splitext(doc.metadata['file_path'])[1].lower()
        is_markdown = (file_extension == ".md")

        # 根据文件类型选择切分器
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter
        logger.info(f'处理文档: {doc.metadata["file_path"]}, 使用切分器: {"Markdown" if is_markdown else "ChineseRecursive"}')

        # 将文档切分成父块  [父块1Document对象, 父块2Document对象]
        parent_docs = parent_splitter_to_use.split_documents([doc]) # split_documents()函数要的是可迭代类型.

        # 遍历每个父块 切成子块
        for j, parent_doc in enumerate(parent_docs):
            parent_id = f"doc_{i}_parent_{j}"
            parent_doc.metadata['parent_id'] = parent_id
            parent_doc.metadata['content'] = parent_doc.page_content
            # 如下输出格式为: 父块: ../data/ai_data\LLM基础知识.pdf, 索引: 9, 唯一ID: doc_0_parent_9
            # print(f'父块: {parent_doc.metadata["file_path"]}, 索引: {j}, 唯一ID: {parent_id}\n\n')
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])

            # 遍历每个子块 -> 添加父块关联信息
            for k, sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata['parent_id'] = parent_id
                sub_chunk.metadata['parent_content'] = parent_doc.page_content
                sub_chunk.metadata['id'] = f"{parent_id}_child_{k}"
                child_chunks.append(sub_chunk)

    # 7. 记录日志.
    logger.info(f"切分完成的子块总数: {len(child_chunks)}")
    # 8. 返回子块列表.
    return child_chunks



if __name__ == "__main__":
    # directory_path = "../data/ai_data"
    # documents = load_documents_from_directory(directory_path)
    # print(f'加载完成的文档总数: {len(documents)}')
    # print(f'前1个文档详情: {documents[0] if documents else "未加载到任何文档"}')

    # 4. 调用文档处理函数 -> 加载指定目录并切分文档.
    chunks = process_documents('../data/ai_data', conf.PARENT_CHUNK_SIZE, conf.CHILD_CHUNK_SIZE, conf.CHUNK_OVERLAP)
    print(f'切分完成的子块总数: {len(chunks)}')
    print(f'前1个子块详情: {chunks[0] if chunks else "未切分到任何子块"}')

