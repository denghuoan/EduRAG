import time

from base import logger
from base.config import config
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.retrieval.bm25_search import BM25Search
from openai import OpenAI
from rag_qa.core.vector_store  import VectorStore
from rag_qa.core.rag_system import RAGSystem

class IntegratedQASystem:
    def __init__(self):
        self.logger = logger
        self.config = config
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
        try:
            self.client = OpenAI(
                api_key=self.config.DASHSCOPE_API_KEY,
                base_url=self.config.DASHSCOPE_BASE_URL
            )

        except Exception as e:
            self.logger.error(f"OpenAI 客户端初始化失败: {e}")
            raise # 抛出异常,终止初始化

        self.vector_store = VectorStore()
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)

    def call_dashscope(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个靠谱的助手, 根据给的信息好好回答问题。"},
                    {"role":"user", "content": prompt },
                ]
            )
            return completion.choices[0].message.content if completion.choices else "错误：无效的 LLM 响应"

        except Exception as e:
            self.logger.error(f"LLM 调用失败: {e}")
            return f"错误：LLM 调用失败 - {e}"

    def query(self, query, source_filter):
        """
        处理用户问题,按步骤来,先查MySQL + BM25 -> 不行再RAG -> 最后返回答案
        :param query:
        :param source_filter:
        :return:
        """
        start_time = time.time()
        self.logger.info(f'开始处理问题: {query}, 来源过滤条件: {source_filter or "不限"}')

        answer,need_rag = self.bm25_search.search(query, threshold=0.85)

        if answer:
            self.logger.info(f"BM25找到了靠谱答案: {answer[:50]}")     # 输出答案可能太长了, 我们只显示前50个字符.
        elif need_rag:
            self.logger.info(f"BM25的答案不靠谱, 让RAG系统来处理...")
            answer = self.rag_system.generate_answer(query, source_filter)
            self.logger.info(f"RAG系统生成了答案: {answer[:50]}")
        else:
            self.logger.info(f'BM25没找到任何答案.')

        # 计算用时.
        processing_time = time.time() - start_time
        # 打印结果.
        self.logger.info(f"处理完成, 用时: {processing_time:.3f}秒")
        return answer if answer else '没有找到和你问题相关的答案, 换个问题试试吧!'


def main ():
    qa_system = IntegratedQASystem()
    qa_system.logger.info('系统初始化好了, 可以开始问问题了!')
    try:
        print('\n------------------------------集成问答系统------------------------------')
        # 打印支持的学科类别
        print(f'支持的来源: {qa_system.config.VALID_SOURCES}')  # ["ai", "java", "test", "ops", "bigdata"]
        # 用法
        print("用法: 输入问题按回车键查看答案, 输入exit按回车键退出系统。")

        while True:
            query=input('\n请录入您的问题').strip()
            if query.lower() == 'exit':
                qa_system.logger.info('用户输入了exit, 准备退出系统...')
                print('拜拜啦, 祝大家一路顺风, 高薪就业!')
                break

            source_filter = input(f'请输入来源过滤(可选, 支持: {"./".join(qa_system.config.VALID_SOURCES)}, 直接回车表示不限)').strip()
            if source_filter:
                if source_filter not in qa_system.config.VALID_SOURCES:
                    qa_system.logger.warning(f'用户输入了无效的来源: {source_filter}')
                    source_filter = None
                else:
                    qa_system.logger.info(f'用户选择了来源过滤: {source_filter}')

            answer = qa_system.query(
                query,
                source_filter=source_filter
            )
            print(f'\n答案: {answer}')

    except Exception as e:
        qa_system.logger.error(f"系统处理异常: {e} 可能需要重启系统")
        print(f'处理问题时出错: {e}, 请联系管理员, 联系电话: 12345678!')
    finally:
        qa_system.mysql_client.close()
        qa_system.logger.info('MySQL连接关闭, 系统退出.')



if __name__ == '__main__':
    main()