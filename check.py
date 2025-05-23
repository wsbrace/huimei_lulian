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
        stats = collection.stats
        row_count = collection.num_entities
        print(f"✅ 集合 {collection_name} 包含 {row_count} 条记录")
        print(f"集合统计信息: {stats}")
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
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            embed_model=embed_model
        )
        print("✅ 向量索引创建成功")
    except Exception as e:
        print(f"❌ 向量索引创建失败: {e}")
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
    elif row_count != len(documents):
        print(f"⚠️ 警告: Milvus集合中的记录数({row_count})与文档数({len(documents)})不匹配!")
    else:
        print(f"✅ 文档成功加载到Milvus中! 文档数: {len(documents)}, Milvus记录数: {row_count}")

if __name__ == "__main__":
    main()