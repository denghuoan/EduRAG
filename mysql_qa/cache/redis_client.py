import os
import sys
import redis
import json

# # todo 1.路径配置
# # 当前脚本所在的文件夹
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 获取当前目录的上一级目录
# module_dir = os.path.dirname(current_dir)
# # 获取项目的根目录
# project_root = os.path.dirname(module_dir)
# # 将项目根目录加入系统路径
# sys.path.insert(0, project_root)

from base import logger,Config

class RedisClient:
    def __init__(self):
        self.logger = logger

        try:
            self.client = redis.StrictRedis(
                host=Config().REDIS_HOST,
                port=Config().REDIS_PORT,
                password=Config().REDIS_PASSWORD,
                db=Config().REDIS_DB,
                decode_responses=True # 自动将返回的字节数据转换为字符串(避免手动解码)
            )
        except redis.RedisError as e:
            self.logger.error(f'Redis连接异常: {e}')
            raise

        self.logger.info('Redis连接成功')

    def set_data(self,key,value):
        try:
            self.client.set(key, json.dumps(value, ensure_ascii=False)) # # ensure_ascii: 保证中文等非ASCII字符正常显式.
            self.logger.info(f'Redis存储数据成功: {key}')
        except redis.RedisError as e:
            self.logger.error(f'Redis存储数据失败: {e}')

    def get_data(self,key):
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f'Redis获取数据失败: {e}')
            return None

    def get_answer(self,query):
        try:
            answer = self.client.get(f'answer:{query}')
            if answer:
                self.logger.info(f'从Redis中获取答案成功: {query}')
                return answer
            # 走到这里, 说明缓存答案不存在, 需要从数据库中查询, 返回None
            return None
        except redis.RedisError as e:
            self.logger.error(f'从Redis中获取答案失败: {e}')
            return None


if __name__ == '__main__':
    redClient = RedisClient()

    # redClient.set_data('if else 能返回上一级继续判断吗','if else  做不到 这样的 逻辑  循环可以')
    # print(redClient.get_answer('if else 能返回上一级继续判断吗'))

    print(redClient.client.keys('*'))