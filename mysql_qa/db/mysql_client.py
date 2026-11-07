import os
import sys

import pandas as pd
import pymysql
from chromadb.utils import results

# todo 1.路径配置
# 当前脚本所在的文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取当前目录的上一级目录
module_dir = os.path.dirname(current_dir)
# 获取项目的根目录
project_root = os.path.dirname(module_dir)
print(project_root)
# 将项目根目录加入系统路径
sys.path.insert(0, project_root)
# 导入自定义模块:配置文件管理模块
from base import Config,logger
# todo 2.定义MySQL客户端
class MySQLClient:
    # todo 2.1 初始化方法 --> 建立数据库连接,创建游标,处理连接异常
    def __init__(self):
        self.logger = logger
        try:
            self.connection = pymysql.connect(
                host=Config().MYSQL_HOST,  # MySQL服务器主机地址
                user=Config().MYSQL_USER,  # MySQL登录用户名
                password=Config().MYSQL_PASSWORD,  # MySQL登录密码
                database=Config().MYSQL_DATABASE  # 数据库名称
            )

            self.cursor = self.connection.cursor()
            self.logger.info("MySQL连接成功")
        except pymysql.Error as e:
            self.logger.error(f'MySQL 连接失败: {e}')
            raise # 抛出异常,不屏蔽错误,让调用方感知错误

    # todo 2.2 创建数据表 --> jpkb,存储:学科知识问答数据
    def create_table(self):
        create_table_query = """
            CREATE TABLE IF NOT EXISTS jpkb (
                id INT AUTO_INCREMENT PRIMARY KEY,
                subject_name VARCHAR(20),
                question VARCHAR(1000),
                answer VARCHAR(1000))
        """

        try:
            # 执行SQL语句
            self.cursor.execute(create_table_query)
            # 提交事务-->确认表创建成功 MySQL默认事务需要手动提交,q确保表创建成功
            self.connection.commit()
            self.logger.info('表创建成功')
        except pymysql.Error as e:
            self.logger.error(f'创建表失败: {e}')
            raise

    # todo 2.3 从csv文件批量插入数据到jpkb中
    def insert_data(self,  csv_path):
        try:
            data = pd.read_csv(csv_path)
            for _, row in data.iterrows():
                # 使用%s占位符, 避免SQL注入攻击问题.
                insert_query = "insert into jpkb (subject_name, question, answer) values (%s, %s, %s)"
                self.cursor.execute(insert_query, (row['学科名称'], row['问题'], row['答案']))

            self.connection.commit()
            self.logger.info("MySQL数据插入成功")

        except pymysql.Error as e:
            self.logger.error(f'数据插入失败: {e}')
            # 事务回滚:把数据库连接恢复到事务开始前的状态
            self.connection.rollback()
            raise

    # todo 2.4 获取所有问题列表
    def fetch_questions(self):
        try:
            self.cursor.execute(
                "select question from jpkb")

            results = self.cursor.fetchall()
            self.logger.info('数据查询成功')
            return results

        except pymysql.Error as e:
            self.logger.error(f'数据查询失败:{e}')
            return []

    # todo 2.5 根据指定问题 从jpkb中获取对应的答案
    def fetch_answer(self, question):
        try:
            self.cursor.execute('select answer from jpkb where question = %s', (question,))
            result = self.cursor.fetchone()
            print(result)
            return result[0] if result else None
        except pymysql.Error as e:
            self.logger.error(f'答案获取失败: {e}')
            return None

    # todo 2.6 关闭MySQL连接 -> 释放资源,避免连接泄漏
    def close(self):
        try:
            self.connection.close()
            self.logger.info("MySQL 连接已关闭")
        except pymysql.Error as e:
            self.logger.error(f"关闭连接失败: {e}")


if __name__ == '__main__':
    mysql_client = MySQLClient()

    # mysql_client.create_table()

    # mysql_client.insert_data(csv_path='../data/JP学科知识问答.csv')

    # questions = mysql_client.fetch_questions()
    # print(questions)

    # answer = mysql_client.fetch_answer('什么是cart算法')
    # print(answer)

    mysql_client.close()



