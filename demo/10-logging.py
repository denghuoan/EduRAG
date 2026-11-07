import logging

# 1. 创建日志记录器
logger = logging.getLogger('Example')
logger.setLevel(logging.DEBUG) # 全局的日志级别
# 对于某个框架，同一个配置存在全局和局部的。绝大部分情况下，局部的配置优先级大于全局的。

# 2. 设置Handler-控制台  用于处理日志的输出位置(控制台,文件),也可以自定义
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) #

# 3. 设置Handle-文件   model = a: append 追加 /  w: write 覆盖
file_handler = logging.FileHandler('../app.log', 'a')
file_handler.setLevel(logging.DEBUG)


# 4. 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 5. 为处理器设置格式
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 6. 将处理器添加到记录器
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 7. 记录日志
logger.debug('调试信息,仅写入文件')
logger.info('程序运行正常')
logger.error('发生错误')

