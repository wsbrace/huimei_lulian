from pymilvus import connections, Collection, utility

# 1. 连接到 Milvus
connections.connect(host="localhost", port="19530")
collection_name = "rag_collection"

# 2. 检查集合是否存在
if not utility.has_collection(collection_name):
    print(f"集合 {collection_name} 不存在！")
    exit()

# 3. 获取集合对象
collection = Collection(collection_name)

# 4. 加载集合（查询前必须加载）
collection.load()

# 5. 查询所有实体
# 使用简单的过滤条件，如 "id >= 0"，获取所有记录
# output_fields 指定要返回的字段，LlamaIndex 默认存储 id、vector 和 metadata
result = collection.query(
    expr="id >= 0",  # 匹配所有实体
    output_fields=["id", "vector", "*"],  # 返回 id、向量和其他元数据字段
    limit=100  # 限制返回数量，避免过多输出
)

# 6. 打印查询结果
print(f"集合 {collection_name} 中共有 {collection.num_entities} 条实体")
for entity in result:
    print("\n实体详情：")
    print(f"ID: {entity['id']}")
    print(f"向量: {entity['vector'][:10]}...")  # 只打印前10个维度
    print(f"元数据: {entity}")  # 打印所有字段，包括 LlamaIndex 存储的文档内容等

# 7. 释放集合
collection.release()