import time
import uuid

import pymysql
from langchain_classic.schema.runnable import history
from openai import OpenAI

from mysql_qa import MySQLClient, BM25Search, RedisClient
from base.config import config
from base.logger import logger
from rag_qa import RAGSystem, VectorStore

class IntegratedQASystem:
    def __init__(self):
        self.logger = logger
        self.config = config
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.init_conversation_table()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

        # 初始化大语言模型
        try:
            self.client = OpenAI(
                api_key= self.config.DASHSCOPE_API_KEY,
                base_url=self.config.DASHSCOPE_BASE_URL,
            )
        except Exception as e:
            self.logger.error(f"OpenAI 客户端初始化失败: {e}")
            raise


        self.vector_store = VectorStore()
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)

        self.init_conversation_table()


    def call_dashscope (self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},
                    { "role": "user", "content": prompt}
                ],
                timeout=30,
                stream=True # 启用流式输出
            )
            for chrunk in completion:
                if chrunk.choices and chrunk.choices[0].delta.content:
                    content = chrunk.choices[0].delta.content
                    yield content

        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            return f"错误：LLM调用失败 - {e}"


    def init_conversation_table(self):
        try:
            self.mysql_client.cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    INDEX idx_session_id (session_id) # 创建索引列, 目的: 提高查询的效率.
                )
            """)
            self.mysql_client.connection.commit()
            self.logger.info("对话历史表初始化成功")

        except pymysql.MySQLError as e:
            self.logger.error(f"初始化对话历史表失败: {e}")
            # 抛出异常，终止初始化
            raise



    def _fetch_recent_history(self,session_id):
        """
        获取最近5轮对话历史
        :param session_id:
        :return: 列表,包含字典形式的对话记录,eg: [{'question':问题, 'answer':答案}]
        """
        try:
            self.mysql_client.cursor.execute("""
                SELECT question,answer
                FROM conversation
                WHERE session_id = %s
                ORDER BY timestamp DESC LIMIT %s
            """, (session_id, 5))
            # 将查询结果转换为字典列表
            result = self.mysql_client.cursor.fetchall()
            history = [{
                "question": row[0],
                "answer": row[1]
            } for row in result]
            return history[::-1] # 反转结果，按时间正序返回

        except pymysql.MySQLError as e:
            self.logger.error(f"获取对话历史失败: {e}")
            # 返回空列表
            return []


    def get_session_history(self,session_id):
        return self._fetch_recent_history(session_id)

    def clear_session_history(self, session_id) -> bool:
        try:
            self.mysql_client.cursor.execute("""
                DELETE
                FROM conversation
                WHERE session_id = %s
            """, (session_id))
            self.mysql_client.connection.commit()
            self.logger.info(f"会话 {session_id} 历史已清除")
            return True

        except pymysql.MySQLError as e:
            self.logger.error(f"清除会话历史失败: {e}")
            self.mysql_client.connection.rollback()
            return False

    # 更新历史会话
    def update_session_history(self,session_id:str, question:str, answer:str):
        try:
            self.mysql_client.cursor.execute("""
                INSERT INTO conversation (session_id, question, answer, timestamp) 
                VALUES (%s, %s, %s, NOW())
            """, (session_id, question, answer))
            history = self._fetch_recent_history(session_id)
            # 删除超出5轮的旧历史
            self.mysql_client.cursor.execute("""
                DELETE
                FROM conversation
                WHERE session_id = %s
                    AND id NOT IN (
                        SELECT id
                        FROM (
                            SELECT id
                            FROM conversation
                            WHERE session_id = %s
                            ORDER BY timestamp DESC LIMIT %s
                             ) AS sub
                    )
            """, (session_id, session_id, 5))

            self.mysql_client.connection.commit()
            self.logger.info(f"会话 {session_id} 历史更新成功")
            return history

        except pymysql.MySQLError as e:
            self.logger.error(f"更新会话历史失败: {e}")
            self.mysql_client.connection.rollback() # 回滚事务
            raise # 抛出异常

        except Exception as e:
            self.logger.error(f"更新会话历史意外错误: {e}")
            self.mysql_client.connection.rollback()
            raise

    # 查询集成系统,支持对话历史和流式输出
    def query(self, query, source_filter=None, session_id=None):
        """

        :param query:
        :param source_filter:
        :param session_id:
        :return: answer生成器  标记是否为完整答案
        """
        start_time = time.time()
        self.logger = logger
        self.logger.info(f"处理查询: '{query}' (会话ID: {session_id})")

        # 获取历史对话: 有session_id-->查询最近5轮对话 没有 --> 返回空列表
        history = self.get_session_history(session_id=session_id) if session_id else []
        print(f'history: {history}')
        answer, need_rag = self.bm25_search.search(query, threshold=0.85)
        # mySQL有可靠答案: 1.更新历史对话
        if answer:
            self.logger.info(f"MySQL答案: {answer}")
            if session_id: # 有si则update  无sid则新建sql
                self.update_session_history(session_id, query, answer)
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            yield answer, True
        elif need_rag:
            self.logger.info("无可靠MySQL答案，回退到RAG")
            collected_answer = ''
            # 从RAG系统获取流式输出
            for token in self.rag_system.generate_answer(query, source_filter, history):
                collected_answer += token
                yield collected_answer, False

            if session_id:
                self.update_session_history(session_id, query, collected_answer)

            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            yield "", True
        else:
            self.logger.info("未找到答案")
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            yield "未找到答案", True



def main():
    qa_system = IntegratedQASystem()
    session_id = str(uuid.uuid4())
    try:
        while True:
            print("\n欢迎使用集成问答系统！")
            print(f"会话ID: {session_id}")
            # 打印支持的学科类别
            print(f"支持的学科类别：{qa_system.config.VALID_SOURCES}")
            # 提示用户输入查询或退出
            print("输入查询进行问答，输入 'exit' 退出。")
            query = input("\n输入查询: ").strip()

            if query.lower() == 'exit':
                logger.info("退出系统")
                print("再见, 感谢您的使用, 期待下次再会！")
                # 退出循环
                break

            source_filter = input(f"请输入学科类别 ({'/'.join(qa_system.config.VALID_SOURCES)}) (直接回车默认不过滤): ").strip()
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                logger.warning(f"无效的学科类别 '{source_filter}'，将不过滤")
                source_filter = None
            print("\n答案: ", end="", flush=True)
            # 生成答案
            answer = ''
            for token, is_complete in qa_system.query(query, source_filter, session_id):
                if token:
                    print(token, end="", flush=True)
                    answer += token
                if is_complete:
                    # 如果是完整答案或流结束，换行并退出循环
                    print()
                    break

            history = qa_system.get_session_history(session_id)
            print("\n最近对话历史:")
            for idx, entry in enumerate(history, 1):
                # 按顺序打印历史记录
                print(f"{idx}. 问: {entry['question']}\n   答: {entry['answer']}")

    except Exception as e:
        logger.error(f"系统错误: {e}")
    finally:
        print('关闭系统')
        qa_system.mysql_client.close()

if __name__ == '__main__':
    main()
    # qa_system = IntegratedQASystem()
    # session_id = str(uuid.uuid4())
    # qa_system.query('熵的解释', session_id)
# 用上下文管理器实现函数运行时间的计算