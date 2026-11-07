# 该脚本用于: RAG系统的核心逻辑 -> 实现RAG(检索增强生成)系统的核心逻辑, 整合查询分类(意图识别), 策略选择, 文档检索 和 答案生成.


# todo 1.导包
import sys, os                                  # 导入路径处理和系统配置.
import time                                     # 导入 time 模块，用于计算时间

from openai.types import vector_store
from rag_qa.core.prompts import RAGPrompts                  # 导入RAG相关的提示模板 -> 定义大模型的输入和输出格式.
from rag_qa.core.query_classifier import QueryClassifier    # 导入查询分类器 -> 判断用户问题属于通用知识还是专业咨询,意图识别
from rag_qa.core.strategy_selector import StrategySelector  # 导入策略选择器 -> 专业咨询的情况下用哪种检索策略(例如:直接检索...)
from rag_qa.core.vector_store import VectorStore            # 导入向量数据库对象 -> 存储和检索文档向量.


# todo 2.配置项目路径 -> 确保程序能正确导入其它目录的模块.
# 1. 获取当前文件的目录的绝对路径.
current_dir = os.path.dirname(os.path.abspath(__file__))        # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa\core
# 2. 获取当前目录的上级目录.
rag_qa_path = os.path.dirname(current_dir)                      # D:\workspace\ai_30_bj\integrated_qa_system\rag_qa
# 3. 将rag_qa目录添加到Python的模块搜索路径.
sys.path.insert(0, rag_qa_path)
# 4. 获取项目根目录.
project_root = os.path.dirname(rag_qa_path)                     # D:\workspace\ai_30_bj\integrated_qa_system
# 5. 将项目根目录添加到Python的模块搜索路径.
sys.path.insert(0, project_root)

from base.config import Config                  # 配置文件
from base.logger import logger                  # 日志对象

# 加载项目配置
conf = Config()

class RAGSystem:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.rag_prompt = RAGPrompts.rag_prompt()
        classifier_path = os.path.join(rag_qa_path, 'models', 'bert_query_classifier')
        self.query_classifier = QueryClassifier(classifier_path)
        self.strategy_selector = StrategySelector()

    # 回溯问题检索
    def _retrieve_with_backtracking(self, query, source_filter=None):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()
        try:
            simplified_query = self.llm(backtrack_prompt_template.format(
                query=query,
            )).strip()
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            # 使用回溯问题进行向量检索,并返回检索结果
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter
            )

        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []

    # 子查询检索
    def _retrieve_with_subqueries(self, query, source_filter=None):
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        subquery_prompt_template = RAGPrompts.subquery_prompt()
        try:
            subqueries_text = self.llm(subquery_prompt_template.format(
                query=query
             )).strip()
            subqueries = [q.strip() for q in subqueries_text.split('\n') if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")

            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return []

            all_docs=[]
            for sub_q in subqueries:
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q,
                    k=conf.RETRIEVAL_K,
                    source_filter=source_filter
                )
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")

            # 去重
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}
            unique_docs = list(unique_docs_dict.values())
            logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            return unique_docs


        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []


    # 假设问题检索
    def _retrieve_with_hyde(self, query, source_filter=None):
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        hyde_prompt_template = RAGPrompts.hyde_prompt()
        try:
            hypo_answer = self.llm(hyde_prompt_template.format(
                query=query
            )).strip()
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter
            )

        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []


    def retrieve_and_merge(self, query, source_filter=None, strategy=None):
        """
        根据指定策略调用对应检索方法
        :param query:
        :param source_filter:
        :param strategy:
        :return:
        """
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        ranked_sub_chunks = []
        if strategy == '回溯问题检索':
            ranked_sub_chunks = self._retrieve_with_backtracking(query, source_filter)
        elif strategy == '子查询检索':
            ranked_sub_chunks = self._retrieve_with_subqueries(query, source_filter)
        elif strategy == '假设问题检索':
            ranked_sub_chunks = self._retrieve_with_hyde(query, source_filter)
        else: # 直接检索
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
                query,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter
            )

        logger.info(f"策略 {strategy} 检索到 {len(ranked_sub_chunks)} 个候选文档")
        final_context_docs = ranked_sub_chunks[: conf.CANDIDATE_M]  # 截取前 conf.CANDIDATE_M 个文档
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")

        return final_context_docs



    def generate_answer(self, query, source_filter=None):
        """
        RAG系统流程: 接收用户查询,自动完成'查询分类' -> 策略选择 -> 文档检索 --> 答案生成
        :param query:
        :param source_filter:
        :return: 最终答案文本
        """
        start_time = time.time()
        logger.info(f"开始处理查询: {query}, 学科过滤: {source_filter}")

        # 意图识别
        query_category = self.query_classifier.predict_category(query)
        if query_category == '通用知识':
            logger.info("查询为通用知识, 直接调用LLM生成答案")
            prompt_input = self.rag_prompt.format(
                context='',
                question=query,
                phone=conf.CUSTOMER_SERVICE_PHONE
            )
            try:
                answer = self.llm(prompt_input)
            except Exception as e:
                logger.error(f'直接调用LLM失败: {e}')
                answer = f"抱歉,处理您的通用知识问题时出错,请联系人工客服: {conf.CUSTOMER_SERVICE_PHONE}"

            processing_time = time.time() - start_time
            logger.info(f"通用知识查询处理完成, 耗时: {processing_time:.3f}s, 查询问题: {query}")
            return answer


        # 专业知识
        logger.info("查询为专业咨询, 执行完整 RAG 流程")
        # 策略选择
        strategy = self.strategy_selector.select_strategy(query)
        # 检索文档
        context_docs = self.retrieve_and_merge(
            query,
            source_filter=source_filter,
            strategy=strategy
        )

        if not context_docs:
            context = ''
            logger.info("未检索到相关文档, 上下文为空")
        else:
            context = "\n\n".join([doc.page_content for doc in context_docs])
            logger.info(f"构建上下文完成, 包含 {len(context_docs)} 个文档块")

        prompt_input = self.rag_prompt.format(
            context=context,
            question=query,
            phone=conf.CUSTOMER_SERVICE_PHONE
        )

        try:
            answer = self.llm(prompt_input)
        except Exception as e:
            logger.error(f'直接调用LLM失败: {e}')
            answer = f"抱歉,处理您的 专业咨询 问题时出错,请联系人工客服: {conf.CUSTOMER_SERVICE_PHONE}"

        processing_time = time.time() - start_time
        logger.info(f"查询处理完成, 耗时: {processing_time:.3f}s, 查询问题: {query}")

        return answer