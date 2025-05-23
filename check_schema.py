"""
用于检查Milvus集合结构的脚本
"""

from pymilvus import connections, Collection, utility

def main():
    # 连接到Milvus
    try:
        print("连接到Milvus...")
        connections.connect("default", host="localhost", port="19530")
        print("✅ Milvus连接成功")
    except Exception as e:
        print(f"❌ Milvus连接失败: {e}")
        return
    
    # 获取所有集合
    try:
        collections = utility.list_collections()
        if not collections:
            print("❌ Milvus中没有集合")
            return
        print(f"Milvus中存在以下集合: {collections}")
    except Exception as e:
        print(f"❌ 获取集合列表失败: {e}")
        return
    
    # 检查指定集合结构
    collection_name = "rag_collection"
    if not utility.has_collection(collection_name):
        print(f"❌ 集合 {collection_name} 不存在")
        return
    
    try:
        # 获取集合信息
        collection = Collection(collection_name)
        
        # 打印集合基本信息
        print(f"\n集合 {collection_name} 的结构信息:")
        
        # 打印所有字段
        print("\n字段信息:")
        for field in collection.schema.fields:
            print(f"字段名: {field.name}")
            print(f"  - 数据类型: {field.dtype}")
            print(f"  - 参数: {field.params}")
            print(f"  - 是否主键: {field.is_primary}")
            print(f"  - 自动ID: {field.auto_id}")
            print()
        
        # 获取记录数
        try:
            collection.load()
            count = collection.num_entities
            print(f"集合包含 {count} 条记录")
            
            if count > 0:
                # 获取一条记录作为示例
                results = collection.query(expr="", output_fields=["*"], limit=1)
                if results:
                    print("\n记录示例:")
                    record = results[0]
                    for key, value in record.items():
                        if isinstance(value, list) and len(value) > 10:
                            print(f"  - {key}: [向量，维度: {len(value)}]")
                        else:
                            print(f"  - {key}: {value}")
                else:
                    print("查询记录失败")
        except Exception as e:
            print(f"❌ 获取记录数失败: {e}")
        
        # 获取索引信息
        try:
            indexes = collection.index()
            print("\n索引信息:")
            for idx in indexes:
                print(f"  字段: {idx.field_name}")
                print(f"  索引类型: {idx.params.get('index_type')}")
                print(f"  距离度量: {idx.params.get('metric_type')}")
                print(f"  构建参数: {idx.params.get('params')}")
                print()
        except Exception as e:
            print(f"❌ 获取索引信息失败: {e}")
    
    except Exception as e:
        print(f"❌ 获取集合结构失败: {e}")

if __name__ == "__main__":
    main() 