import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

# 调试函数
def check_milvus_connection():
    """检查Milvus连接状态"""
    try:
        from pymilvus import connections
        connections.connect("default", host="localhost", port="19530")
        print("✅ Milvus连接成功")
        return True
    except Exception as e:
        print(f"❌ Milvus连接失败: {e}")
        return False

def check_collection_exists(collection_name):
    """检查集合是否存在"""
    try:
        from pymilvus import utility
        exists = utility.has_collection(collection_name)
        print(f"{'✅' if exists else '❌'} 集合 {collection_name} {'存在' if exists else '不存在'}")
        return exists
    except Exception as e:
        print(f"❌ 检查集合存在失败: {e}")
        return False

def get_collection_stats(collection_name):
    """获取集合统计信息"""
    try:
        from pymilvus import Collection
        collection = Collection(collection_name)
        collection.load()
        row_count = collection.num_entities
        print(f"✅ 集合 {collection_name} 包含 {row_count} 条记录")
        
        # 尝试查询一些记录以确认数据存在
        if row_count > 0:
            print("尝试查询部分记录...")
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = collection.query(expr="id >= 0", limit=3, output_fields=["id"])
            if results:
                print(f"查询到记录: {results}")
            else:
                print("⚠️ 警告: 查询返回空结果")
        return row_count
    except Exception as e:
        print(f"❌ 获取集合统计失败: {e}")
        return 0

def get_embedding_model():
    """获取可用的embedding模型"""
    embedding_dim = 0
    
    # 尝试使用阿里云文心向量模型
    try:
        print("正在配置阿里云文心向量模型...")
        # 检查API密钥是否存在
        if not os.getenv("DASHSCOPE_API_KEY"):
            print("⚠️ 警告: 未设置DASHSCOPE_API_KEY环境变量")
            raise ValueError("需要设置DASHSCOPE_API_KEY环境变量")
            
        # 使用阿里云文心向量模型
        embed_model = DashScopeEmbedding(
            model_name="text-embedding-v2",  # 阿里云文心向量模型
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        embedding_dim = 1536  # 文心向量模型维度是1536
        print("✅ 阿里云文心向量模型配置成功")
        return embed_model, embedding_dim
    except Exception as e:
        print(f"❌ 阿里云embedding配置失败: {e}")
    
    # 尝试使用OpenAI向量模型
    try:
        print("尝试使用OpenAI向量模型...")
        # 检查API密钥是否存在
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️ 警告: 未设置OPENAI_API_KEY环境变量")
            raise ValueError("需要设置OPENAI_API_KEY环境变量")
            
        # 使用OpenAI向量模型
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",  # OpenAI向量模型
            api_key=os.getenv("OPENAI_API_KEY")
        )
        embedding_dim = 1536  # OpenAI向量模型维度是1536
        print("✅ OpenAI向量模型配置成功")
        return embed_model, embedding_dim
    except Exception as e2:
        print(f"❌ OpenAI embedding配置失败: {e2}")
    
    # 作为最后尝试，使用本地模型（如果可用）
    try:
        print("尝试使用本地向量模型...")
        # 使用sentence-transformers模型，这个更常见且更容易在本地下载
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 支持中文的多语言模型
        )
        embedding_dim = 384  # 此模型的维度
        print("✅ 本地向量模型配置成功")
        return embed_model, embedding_dim
    except Exception as e3:
        print(f"❌ 本地embedding配置失败: {e3}")
    
    print("❌ 所有embedding模型配置失败")
    return None, 0

def main():
    # 1. 配置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("检查环境变量...")
    print(f"- DASHSCOPE_API_KEY: {'已设置' if os.getenv('DASHSCOPE_API_KEY') else '未设置'}")
    print(f"- OPENAI_API_KEY: {'已设置' if os.getenv('OPENAI_API_KEY') else '未设置'}")

    # 2. 检查数据目录
    data_dir = "./data"
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录 {data_dir} 不存在!")
        return
    else:
        files = os.listdir(data_dir)
        print(f"✅ 数据目录 {data_dir} 存在, 包含 {len(files)} 个文件:")
        for file in files:
            print(f"  - {file} ({os.path.getsize(os.path.join(data_dir, file))} 字节)")

    # 3. 加载文档
    try:
        documents = SimpleDirectoryReader(data_dir).load_data()
        print(f"✅ 成功加载 {len(documents)} 个文档")
        for i, doc in enumerate(documents[:3]):  # 打印前3个文档的摘要
            print(f"  文档 {i+1}: 内容长度 {len(doc.text)} 字符, 开头: {doc.text[:50]}...")
        if len(documents) > 3:
            print(f"  ... 还有 {len(documents)-3} 个文档")
    except Exception as e:
        print(f"❌ 加载文档失败: {e}")
        return

    # 4. 检查Milvus连接
    if not check_milvus_connection():
        print("请确保Milvus服务正在运行")
        return

    # 5. 获取embedding模型
    embed_model, embedding_dim = get_embedding_model()
    if embed_model is None:
        print("无法找到可用的embedding模型，退出程序")
        return

    # 6. 配置Milvus向量存储
    collection_name = "rag_collection"
    print(f"检查集合 {collection_name} 是否已存在")
    check_collection_exists(collection_name)

    print("配置向量存储...")
    try:
        vector_store = MilvusVectorStore(
            uri="http://localhost:19530",
            collection_name=collection_name,
            dim=embedding_dim,
            overwrite=True
        )
        print("✅ 向量存储配置成功")
    except Exception as e:
        print(f"❌ 向量存储配置失败: {e}")
        return

    # 7. 创建向量索引
    print("正在创建向量索引...")
    try:
        # 检查文档是否为空
        if not documents:
            print("❌ 错误: 文档列表为空!")
            return
        
        print(f"开始向量索引创建过程，文档数量: {len(documents)}")
        
        # 打印一些调试信息
        print("正在为文档嵌入向量并存储到Milvus...")
        
        # 手动尝试对一个文档进行嵌入以测试
        try:
            print("测试单个文档嵌入...")
            test_text = documents[0].text[:100]  # 取第一个文档的前100个字符
            print(f"文档片段: {test_text}")
            test_embedding = embed_model.get_text_embedding(test_text)
            print(f"生成的嵌入向量维度: {len(test_embedding)}, 示例值: {test_embedding[:5]}...")
            print("✅ 嵌入测试成功")
        except Exception as embed_err:
            print(f"❌ 嵌入测试失败: {embed_err}")
        
        # 创建向量索引
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            embed_model=embed_model,
            show_progress=True  # 显示进度
        )
        print("✅ 向量索引创建成功")
    except Exception as e:
        print(f"❌ 向量索引创建失败: {e}")
        import traceback
        traceback.print_exc()  # 打印完整的错误堆栈
        return

    # 8. 检查集合统计信息
    print("检查Milvus集合中的记录数...")
    row_count = get_collection_stats(collection_name)
    if row_count == 0:
        print("⚠️ 警告: Milvus集合中没有记录!")
        print("可能的原因:")
        print("  1. documents列表为空")
        print("  2. 向量嵌入过程失败")
        print("  3. 将嵌入向量存入Milvus失败")
        
        # 尝试直接插入一条测试数据到Milvus
        try:
            print("\n尝试直接插入测试数据到Milvus...")
            from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
            
            # 如果集合不存在，创建集合
            if not check_collection_exists(collection_name):
                print(f"创建测试集合 {collection_name}")
                # 创建包含id和doc_id两个字段的集合
                id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True, auto_id=False)  # 主键id字段
                doc_id_field = FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100)  # 额外的doc_id字段
                vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
                text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
                schema = CollectionSchema(fields=[id_field, doc_id_field, vector_field, text_field], description="测试集合")
                collection = Collection(name=collection_name, schema=schema)
            else:
                collection = Collection(collection_name)
                
            # 获取集合的字段信息
            print("集合字段信息:")
            for field in collection.schema.fields:
                print(f"  字段: {field.name}, 类型: {field.dtype}, 参数: {field.params}")
                
            # 生成测试向量
            test_text = "这是一条测试文本"
            test_vector = embed_model.get_text_embedding(test_text)
            
            # 插入数据，根据集合结构提供必要的字段
            insert_data = {
                "doc_id": "test_id_001", 
                "vector": test_vector, 
                "text": test_text
            }
            
            # 检查字段名称，同时添加id字段
            if any(field.name == "id" for field in collection.schema.fields):
                insert_data["id"] = "test_primary_id_001"
                
            print(f"插入数据: {insert_data.keys()}")
            collection.insert([insert_data])
            
            # 刷新以确保数据可见
            collection.flush()
            
            # 获取集合信息
            print("✅ 测试数据插入成功")
            print(f"集合现在包含 {collection.num_entities} 条记录")
            
            # 查询验证
            try:
                query_fields = ["text"]
                if any(field.name == "doc_id" for field in collection.schema.fields):
                    query_fields.append("doc_id")
                if any(field.name == "id" for field in collection.schema.fields):
                    query_fields.append("id")
                    
                results = collection.query(expr="", output_fields=query_fields, limit=3)
                if results:
                    print(f"查询到 {len(results)} 条记录:")
                    for idx, r in enumerate(results):
                        print(f"记录 {idx+1}:")
                        if "id" in r:
                            print(f"  ID: {r.get('id', 'N/A')}")
                        if "doc_id" in r:
                            print(f"  doc_id: {r.get('doc_id', 'N/A')}")
                        text = r.get('text', 'N/A')
                        print(f"  文本: {text[:50]}...")
                else:
                    print("查询结果为空")
            except Exception as query_err:
                print(f"查询失败: {query_err}")
                
        except Exception as insert_err:
            print(f"❌ 测试数据插入失败: {insert_err}")
            
    elif row_count != len(documents):
        print(f"⚠️ 警告: Milvus集合中的记录数({row_count})与文档数({len(documents)})不匹配!")
    else:
        print(f"✅ 文档成功加载到Milvus中! 文档数: {len(documents)}, Milvus记录数: {row_count}")

if __name__ == "__main__":
    main()