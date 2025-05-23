"""
直接导入数据到Milvus的脚本，绕过llama-index
"""

import os
import sys
import time
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.dashscope import DashScopeEmbedding

# 连接到Milvus
def connect_milvus():
    try:
        print("连接到Milvus...")
        connections.connect("default", host="localhost", port="19530")
        print("✅ Milvus连接成功")
        return True
    except Exception as e:
        print(f"❌ Milvus连接失败: {e}")
        return False

# 获取embedding模型
def get_embedding_model():
    # 首先尝试文心大模型
    try:
        if os.getenv("DASHSCOPE_API_KEY"):
            print("正在加载文心embedding模型...")
            model = DashScopeEmbedding(
                model_name="text-embedding-v2",
                api_key=os.getenv("DASHSCOPE_API_KEY")
            )
            print("✅ 文心embedding模型加载成功")
            return model, 1536  # 文心向量维度
    except Exception as e:
        print(f"❌ 文心embedding模型加载失败: {e}")

    # 然后尝试本地模型
    try:
        print("正在加载本地embedding模型...")
        model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        print("✅ 本地embedding模型加载成功")
        return model, 384  # 本地模型向量维度
    except Exception as e:
        print(f"❌ 本地embedding模型加载失败: {e}")
    
    return None, 0

# 创建或获取集合
def get_collection(collection_name, dim):
    try:
        # 检查集合是否存在
        if utility.has_collection(collection_name):
            print(f"集合 {collection_name} 已存在，将重新创建")
            utility.drop_collection(collection_name)
        
        # 创建集合
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields=fields, description="文档集合")
        collection = Collection(name=collection_name, schema=schema)
        
        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        print(f"✅ 集合 {collection_name} 创建成功，向量维度: {dim}")
        return collection
    except Exception as e:
        print(f"❌ 创建集合失败: {e}")
        return None

# 加载文档
def load_documents(data_dir):
    try:
        print(f"从 {data_dir} 加载文档...")
        if not os.path.exists(data_dir):
            print(f"❌ 数据目录 {data_dir} 不存在")
            return []
            
        files = os.listdir(data_dir)
        if not files:
            print(f"❌ 数据目录 {data_dir} 为空")
            return []
            
        documents = SimpleDirectoryReader(data_dir).load_data()
        print(f"✅ 成功加载 {len(documents)} 个文档")
        return documents
    except Exception as e:
        print(f"❌ 加载文档失败: {e}")
        return []

# 对文档生成向量并插入Milvus
def import_documents(documents, collection, embed_model):
    if not documents:
        print("没有文档可导入")
        return 0
        
    print(f"开始导入 {len(documents)} 个文档到Milvus...")
    batch_size = 10  # 每次批量处理的文档数
    total_imported = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        texts = [doc.text for doc in batch]
        
        print(f"处理批次 {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        # 生成向量
        vectors = []
        for text in texts:
            try:
                vector = embed_model.get_text_embedding(text)
                vectors.append(vector)
            except Exception as e:
                print(f"❌ 生成向量失败: {e}")
                vectors.append([0] * embed_model.get_text_embedding_dimension())  # 填充零向量
        
        # 准备插入数据
        entities = [
            {"text": texts[j], "vector": vectors[j]}
            for j in range(len(texts))
        ]
        
        # 插入Milvus
        try:
            result = collection.insert(entities)
            insert_count = len(result.primary_keys)
            total_imported += insert_count
            print(f"✅ 成功插入 {insert_count} 条记录，总计: {total_imported}")
        except Exception as e:
            print(f"❌ 插入数据失败: {e}")
    
    # 刷新集合以确保数据可见
    collection.flush()
    print(f"导入完成，总共导入 {total_imported} 条记录")
    return total_imported

def main():
    # 1. 连接Milvus
    if not connect_milvus():
        return
    
    # 2. 加载embedding模型
    embed_model, dim = get_embedding_model()
    if not embed_model:
        print("无法加载embedding模型，退出")
        return
    
    # 3. 创建集合
    collection_name = "rag_collection"
    collection = get_collection(collection_name, dim)
    if not collection:
        return
    
    # 4. 加载文档
    data_dir = "./data"
    documents = load_documents(data_dir)
    if not documents:
        return
    
    # 5. 导入文档
    imported_count = import_documents(documents, collection, embed_model)
    
    # 6. 验证导入结果
    collection.load()
    count = collection.num_entities
    print(f"\n验证结果: 集合 {collection_name} 现有 {count} 条记录")
    if count > 0:
        print("\n尝试查询部分记录...")
        results = collection.query(expr="id >= 0", output_fields=["id", "text"], limit=3)
        if results:
            print(f"查询到 {len(results)} 条记录:")
            for i, r in enumerate(results):
                print(f"记录 {i+1}:")
                print(f"  ID: {r.get('id', 'N/A')}")
                text = r.get('text', 'N/A')
                if isinstance(text, str) and len(text) > 100:
                    text = text[:100] + "..."
                print(f"  文本: {text}")
        else:
            print("查询结果为空")
    else:
        print("集合中没有记录，导入失败")

if __name__ == "__main__":
    main() 