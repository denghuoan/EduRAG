from utils.logger import *

logger = setup_logger('AI')

def process_data(data:{upper}):

    logger.debug(f'开始处理数据:{data}')

    if not data:
        logger.error('数据为空,无法处理')
        return None

    logger.info('数据处理完成')

    return data.upper()


def main():
    # 1. 记录INFO级日志, 提示程序开始运行.
    logger.info('程序开始运行')

    # 2. 调用数据处理函数, 处理数据.
    # result = process_data('')
    result = process_data('马上端午节了, 祝大家端午安康! qy')

    # 3. 根据处理结果输出不同日志.
    if result:
        # 3.1 若结果非空 -> 处理成功, 记录INFO日志.
        logger.info(f'处理结果: {result}')
    else:
        # 3.2 若结果为空 -> 处理失败, 记录WARNING日志.
        logger.warning('处理失败')

    # 4. 记录INFO级日志, 提示程序结束.
    logger.info('程序结束')



# todo 4. 测试代码 -> 程序的主入口, 触发程序执行.
if __name__ == '__main__':
    main()