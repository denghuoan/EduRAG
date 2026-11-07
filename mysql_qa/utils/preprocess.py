import jieba
from base import logger

def preprocess_text(text):
    try:
        return jieba.lcut(text.lower())
    except AttributeError as e:
        logger.error(f"文本预处理失败: {e}")
        return []


if __name__ == '__main__':
    print(preprocess_text('黑马程序员'))