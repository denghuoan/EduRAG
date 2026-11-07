from base import logger, Config
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.utils.preprocess import preprocess_text
from rank_bm25 import BM25Okapi
import numpy as np

class BM25Search:
    # todo 1.1 初始化函数
    def __init__(self, redis_client, mysql_client):

        self.logger = logger
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        self.bm25 = None
        self.question = None
        self.original_questions = None
        self._load_data()

    # todo 1.2 数据加载方法: 优先Redis --> MySQL
    def _load_data(self):
        # 1.定义Redis缓存
        original_key = "qa_original_questions" # 原始问题缓存键 [问题1, 问题2, 问题3,...]
        tokenized_key = "qa_tokenized_questions" # 分词后问题缓存建 [[问题1分词1, 词1, 词3], [问题2分词1, 词1, 词3]
        # 2. 从redis中获取原始问题
        self.original_questions = self.redis_client.get_data(original_key)
        print(f'Redis中所有问题:{self.original_questions}')
        # 3.从redis获取分词后的问题
        tokenized_questions =  self.redis_client.get_data(tokenized_key)

        # 4.若redis缓存中无数据 -> 则从MySQL加载并更新缓存
        if not self.original_questions or not tokenized_questions:
            self.original_questions = self.mysql_client.fetch_questions() # 获取所有问题列表
            print(f'MySQL中所有问题:{self.original_questions}')
            # MySQL中无问题数据
            if not self.original_questions: # MySQL中也无问题数据
                self.logger.warning('未从MySQL加载到任何问题数据')
                return

            # MySQL中有问题: 1.分词处理 2.将原始问题缓存到redis中
            tokenized_questions = [preprocess_text(q[0]) for q in self.original_questions] # 二维数组,每条数据的分词列表
            self.redis_client.set_data(original_key, [(q[0]) for q in self.original_questions])
            self.redis_client.set_data(tokenized_key, tokenized_questions)

        # redis中有数据 --> 初始化BM25模型
        self.questions = tokenized_questions
        self.bm25 = BM25Okapi(self.questions)
        self.logger.info('BM25模型 初始化成功')

    # softmax分数归一化方法: 将BM25分数转换为概率分布便于阈值判断
    def _softmax(self, scores):
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)

    def search(self, query, threshold=0.85):
        """
        根据输入查询检索 最相似的问题 并返回对应答案
        :return: 匹配成功: (答案,False)  未匹配: (None, True)  False意思是: 非新查询(命中缓存)
        """
        # 1.检查查询有效性
        if not query or not isinstance(query, str):
            self.logger.error("无效查询: 查询为空或者非字符串类型")
            return None, True

        # 2. 检查Redis缓存
        cached_answer = self.redis_client.get_answer(query)
        print(f'cached_answer: {cached_answer}')
        if cached_answer:
            return cached_answer, False

        # 3. 缓存未命中,需要从MySQL查询
        try:
            # 对查询文本进行预处理
            query_tokens = preprocess_text(query)
            scores = self.bm25.get_scores(query_tokens)
            softmax_scores = self._softmax(scores)
            best_idx = softmax_scores.argmax() # 最高分索引
            best_score = softmax_scores[best_idx]
            print(f'best_idx: {best_idx}, best_score: {best_score}')


            # 若最高分超过阈值,视为匹配成功
            if best_score >= threshold:
                original_question = self.original_questions[best_idx]
                answer = self.mysql_client.fetch_answer(original_question)
                if answer:
                    self.redis_client.set_data(f'answer:{query}', answer)
                    self.logger.info(f'搜索成功, 相似度: {best_score:.3f}')
                    return answer, False

            # 未超过阈值
            self.logger.info(f'未找到可靠答案, 最高匹配度: {best_score:.3f}(低于阈值{threshold:.3f})')
            return None, True

        except Exception as e:
            self.logger.error(f'数据库查询异常: {e}')
            return None, True


if __name__ == "__main__":
    redis_client = RedisClient()
    mysql_client = MySQLClient()
    bm25_search = BM25Search(redis_client, mysql_client)
    result = bm25_search.search(query = 'if else 能返回上一级继续判断吗')
    print(result)