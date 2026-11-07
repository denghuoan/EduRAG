import logging
import os


# 1.计算日志文件的最终保存路径
# 获取当前脚本文件的绝对路径
module_path = os.path.abspath(__file__)
# 获取当前脚本所在目录的路径
base_path = os.path.dirname(module_path)
# 获取当前脚本所在目录的父目录的路径
current_path = os.path.dirname(base_path)
print(current_path)
# 拼接日志文件的完整路径:  上级目录+logs文件夹+app.log文件
log_file = os.path.join(current_path, "logs/app.log")
print(log_file)
# 2.封装日志配置函数 -> 可重复使用的双输出日志
def setup_logger(name, log_file: str|bytes=log_file):
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # 创建日志输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

if __name__ == '__main__':
    logger = setup_logger("test")