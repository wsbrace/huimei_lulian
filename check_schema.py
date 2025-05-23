"""
检查Milvus集合结构的工具
"""

import os
from pymilvus import connections, Collection, utility

def main():
    # 连接到Milvus
    print("连接到Milvus...")
    connections.connect("default", host="localhost", port="19530")
    print("✅ Milvus连接成功")
    
    # 获取所有集合
    collections = utility.list_collections()
    print(f"Milvus中的集合: {collections}")
    
    # 如果没有指定集合名称，则使用默认的rag_collection
    collection_name = "rag_collection"
    
    # 检查集合是否存在
    if not utility.has_collection(collection_name):
        print(f"❌ 集合 {collection_name} 不存在")
        return
    
    # 打印集合详细信息
    collection = Collection(collection_name)
    print(f"\n集合名称: {collection_name}")
    print(f"记录数量: {collection.num_entities}")
    print("\n字段信息:")
    
    # 获取并打印每个字段的详细信息
    for field in collection.schema.fields:
        print(f"  字段名: {field.name}")
        print(f"  字段类型: {field.dtype}")
        print(f"  字段参数: {field.params}")
        if hasattr(field, "is_primary") and field.is_primary:
            print(f"  是主键: 是")
        print("")
    
    # 打印索引信息
    print("\n索引信息:")
    for index in collection.indexes:
        print(f"  索引字段: {index.field_name}")
        print(f"  索引参数: {index.params}")
        print("")
    
    # 尝试查询一些数据
    if collection.num_entities > 0:
        print("\n尝试查询数据...")
        collection.load()
        
        # 获取所有字段名
        field_names = [field.name for field in collection.schema.fields 
                     if field.dtype != 101]  # 排除向量字段
        
        try:
            results = collection.query(expr="", output_fields=field_names, limit=3)
            if results:
                print(f"查询到 {len(results)} 条记录:")
                for i, record in enumerate(results):
                    print(f"\n记录 {i+1}:")
                    for field in field_names:
                        value = record.get(field, "N/A")
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + "..."
                        print(f"  {field}: {value}")
            else:
                print("查询结果为空")
        except Exception as e:
            print(f"查询失败: {e}")
    
    print("\n检查完成！")

if __name__ == "__main__":
    main() 