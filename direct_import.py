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
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True, auto_id=False),  # 使用VARCHAR类型
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),  # 额外的doc_id字段
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)  # 向量字段名为embedding
        ]
        schema = CollectionSchema(fields=fields, description="文档集合")
        collection = Collection(name=collection_name, schema=schema)
        
        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="embedding", index_params=index_params)  # 索引embedding字段
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
            {"id": f"id_{i+j:06d}", "doc_id": f"doc_{i+j:06d}", "text": texts[j], "embedding": vectors[j]}  # 使用embedding字段名
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

# 为LlamaIndex创建一个特殊的导入方法
def import_for_llamaindex(data_dir, collection_name, embed_model):
    """为LlamaIndex创建集合并导入数据，确保ID字段正确"""
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
    from llama_index.core.schema import TextNode
    
    print("\n专为LlamaIndex创建集合并导入数据...")
    
    # 1. 连接到Milvus
    connections.connect("default", host="localhost", port="19530")
    
    # 2. 检查并创建集合
    dim = embed_model.get_text_embedding_dimension()
    
    # 如果集合存在，先删除
    if utility.has_collection(collection_name):
        print(f"删除已存在的集合 {collection_name}")
        utility.drop_collection(collection_name)
    
    # 创建集合字段
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True, auto_id=False),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description="文档集合")
    collection = Collection(name=collection_name, schema=schema)
    
    # 创建索引
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"✅ 创建集合成功，向量维度: {dim}")
    
    # 3. 加载文档
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录 {data_dir} 不存在")
        return
    
    documents = SimpleDirectoryReader(data_dir).load_data()
    print(f"✅ 加载了 {len(documents)} 个文档")
    
    # 4. 创建节点
    nodes = []
    for i, doc in enumerate(documents):
        node_id = f"doc_{i:06d}"
        node = TextNode(
            text=doc.text,
            id_=node_id,
            metadata={"doc_id": node_id}
        )
        nodes.append(node)
    
    print(f"创建了 {len(nodes)} 个文档节点")
    
    # 5. 生成嵌入向量和插入数据
    batch_size = 5
    total_inserted = 0
    
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i+batch_size]
        print(f"处理批次 {i//batch_size + 1}/{(len(nodes)-1)//batch_size + 1}")
        
        # 生成嵌入向量
        entities = []
        for j, node in enumerate(batch):
            try:
                vector = embed_model.get_text_embedding(node.text)
                entities.append({
                    "id": node.id_,
                    "doc_id": node.metadata.get("doc_id", node.id_),
                    "text": node.text,
                    "embedding": vector
                })
                print(f"  处理节点 {i+j+1}/{len(nodes)}: id={node.id_}")
            except Exception as e:
                print(f"❌ 生成嵌入向量失败: {e}")
        
        # 插入数据
        try:
            result = collection.insert(entities)
            insert_count = len(result.primary_keys)
            total_inserted += insert_count
            print(f"✅ 批次 {i//batch_size + 1} 成功插入 {insert_count} 条记录")
        except Exception as e:
            print(f"❌ 插入数据失败: {e}")
    
    # 刷新集合
    collection.flush()
    print(f"✅ 总共向Milvus插入 {total_inserted} 条记录")
    
    # 6. 验证插入结果
    try:
        collection.load()
        count = collection.num_entities
        print(f"✅ 集合 {collection_name} 中有 {count} 条记录")
        
        if count > 0:
            print("\n查询示例记录:")
            results = collection.query(expr="", output_fields=["id", "doc_id", "text"], limit=1)
            if results:
                for i, r in enumerate(results):
                    print(f"记录 {i+1}:")
                    print(f"  ID: {r.get('id', 'N/A')}")
                    print(f"  doc_id: {r.get('doc_id', 'N/A')}")
                    text = r.get('text', 'N/A')
                    if len(text) > 100:
                        text = text[:100] + "..."
                    print(f"  文本: {text}")
            else:
                print("查询结果为空")
    except Exception as e:
        print(f"❌ 验证插入结果失败: {e}")
    
    return total_inserted

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
    
    # 4. 使用LlamaIndex兼容方式导入数据
    data_dir = "./data"
    import_for_llamaindex(data_dir, collection_name, embed_model)
    
    print("\n✅ 所有操作完成！")

if __name__ == "__main__":
    main() 