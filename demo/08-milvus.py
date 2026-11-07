from accelerate.commands.config.update import description
from pymilvus import MilvusClient,DataType,AnnSearchRequest,WeightedRanker

# 1. 配置数据库
def operate_db():
    client = MilvusClient(url='http://localhost:19530')

    database = client.list_databases()

    if "milvus_demo" not in database:
        client.create_database(db_name='milvus_demo')
        print('创建成功')
    else:
        client.use_database(db_name = 'milvus_demo')
        print('使用 milvus_demo 成功')

    return client


mClient = operate_db()


def operate_collection():
    # 1.定义集合的Schema(相当于:表结构)
    #   schema用于规定集合中包含哪些字段,各字段的数据类型和约束条件
    #       auto_id=False主键字段需要手动赋值,不自动生成
    #       enable_dynamic_field=True允许集合中添加动态字段,即:不需要提前定义字段,这些动态添加的字段会以json格式存在$meta中
    schema = mClient.create_schema(auto_id=False, enable_dynamic_field=True)

    # 2.添加字段
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=5)
    schema.add_field(field_name="scalar", datatype=DataType.VARCHAR, max_length=256,description='标量字段')

    # 3.创建集合
    mClient.create_collection(collection_name='demo_v1', schema=schema)
    print('demo_v1集合创建成功')


def operate_index():
    # 场景一:为向量字段添加索引
    # 1. 初始化索引配置对象,用于设置索引相关参数
    prepare_indexs = mClient.prepare_index_params()
    # 2. 配置向量字段的索引信息
    prepare_indexs.add_index(field_name='vector', index_type='', metric_type='COSINE', index_name='vector_index')
    # 3. 执行索引创建操作
    mClient.create_index(collection_name='demo_v1', index_params=prepare_indexs)

    res = mClient.list_indexes(collection_name='demo_v1')
    print(res)

if __name__ == '__main__':
    # operate_collection()
    operate_index()