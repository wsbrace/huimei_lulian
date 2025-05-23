"""
Milvus数据检查工具，用于查看Milvus中是否有数据
"""

import os
from pymilvus import connections, Collection, utility

def main():
    # 1. 连接到Milvus
    try:
        print("正在连接到Milvus...")
        connections.connect(host="localhost", port="19530")
        print("✅ Milvus连接成功")
    except Exception as e:
        print(f"❌ Milvus连接失败: {e}")
        return

    # 2. 列出所有集合
    try:
        collections = utility.list_collections()
        if collections:
            print(f"✅ Milvus中存在以下集合: {collections}")
        else:
            print("⚠️ Milvus中没有集合")
            return
    except Exception as e:
        print(f"❌ 获取集合列表失败: {e}")
        return

    # 3. 选择要查询的集合
    collection_name = "rag_collection"
    
    # 4. 检查集合是否存在
    if not utility.has_collection(collection_name):
        print(f"❌ 集合 {collection_name} 不存在")
        return
    
    # 5. 获取集合信息
    collection = Collection(collection_name)
    collection.load()
    
    # 检查集合字段
    print("\n集合字段信息:")
    for field in collection.schema.fields:
        print(f"字段: {field.name}, 类型: {field.dtype}, 参数: {field.params}")
    
    # 6. 获取记录数
    try:
        count = collection.num_entities
        print(f"\n✅ 集合 {collection_name} 包含 {count} 条记录")
        
        if count == 0:
            print("⚠️ 警告: 集合中没有数据，请检查索引创建过程")
            return
            
        # 7. 查询部分记录
        print("\n正在查询部分记录...")
        try:
            results = collection.query(
                expr="doc_id != ''",  # 匹配所有非空doc_id
                output_fields=["doc_id", "text"],  # 只返回doc_id和text字段
                limit=3  # 限制返回3条记录
            )
            
            if results:
                print(f"✅ 查询到 {len(results)} 条记录:")
                for idx, r in enumerate(results):
                    print(f"\n记录 {idx+1}:")
                    print(f"  ID: {r.get('doc_id', 'N/A')}")
                    text = r.get('text', 'N/A')
                    if isinstance(text, str) and len(text) > 100:
                        text = text[:100] + "..."
                    print(f"  文本: {text}")
            else:
                print("⚠️ 警告: 查询结果为空")
        except Exception as query_err:
            print(f"❌ 查询失败: {query_err}")
            print("尝试直接列出集合中的记录...")
            try:
                # 如果查询失败，尝试使用不同的方法查询
                import random
                import time
                query_id = f"query_{int(time.time())}"
                collection.create_index(field_name="doc_id", index_params={"index_type": "FLAT"})
                results = collection.query(expr="", output_fields=["doc_id", "text"], limit=3)
                if results:
                    print(f"✅ 获取到 {len(results)} 条记录")
                    for idx, r in enumerate(results):
                        print(f"记录 {idx+1}: {r}")
                else:
                    print("⚠️ 获取记录为空")
            except Exception as e2:
                print(f"❌ 再次查询失败: {e2}")
        
        # 8. 尝试进行向量搜索
        print("\n尝试进行向量搜索(随机向量)...")
        import random
        import numpy as np
        
        # 获取向量维度
        vector_field = None
        for field in collection.schema.fields:
            if field.dtype == 101:  # FLOAT_VECTOR类型
                vector_field = field.name
                vector_dim = field.params.get("dim")
                break
        
        if vector_field and vector_dim:
            print(f"找到向量字段: {vector_field}，维度: {vector_dim}")
            # 生成随机向量
            random_vector = np.random.random([vector_dim]).tolist()
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            try:
                results = collection.search(
                    data=[random_vector],
                    anns_field="embedding",
                    param=search_params,
                    limit=3,
                    output_fields=["doc_id", "text"]
                )
                
                if results and results[0]:
                    print(f"✅ 搜索到 {len(results[0])} 条记录:")
                    for idx, hit in enumerate(results[0]):
                        print(f"\n结果 {idx+1} (距离: {hit.distance}):")
                        print(f"  ID: {hit.entity.get('doc_id', 'N/A')}")
                        text = hit.entity.get('text', 'N/A')
                        if isinstance(text, str) and len(text) > 100:
                            text = text[:100] + "..."
                        print(f"  文本: {text}")
                else:
                    print("⚠️ 警告: 搜索结果为空")
            except Exception as e:
                print(f"❌ 向量搜索失败: {e}")
        else:
            print(f"❌ 未找到向量字段，可能集合Schema有问题")
    
    except Exception as e:
        print(f"❌ 查询集合信息失败: {e}")
        return

if __name__ == "__main__":
    main() 