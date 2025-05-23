import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import OpenAI
from typing import Any, Optional
from pydantic import BaseModel, Field
import pymilvus  # 添加pymilvus导入用于检查
# 引入更多embedding选项
from llama_index.embeddings.dashscope import DashScopeEmbedding  # 阿里云文心embedding
from llama_index.embeddings.openai import OpenAIEmbedding  # OpenAI embedding

# 添加调试函数
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

# Custom LLM for Qianwen API
class QianwenLLM(CustomLLM, BaseModel):
    model_name: str = Field(default="qwen-plus", description="Qianwen model name")
    api_key: Optional[str] = Field(default=None, description="API key for Qianwen")
    api_base: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", description="Qianwen API base URL")
    client: Any = Field(default=None, description="OpenAI client for Qianwen API")

    def __init__(self, **data):
        super().__init__(**data)
        self.client = OpenAI(api_key=self.api_key or os.getenv("DASHSCOPE_API_KEY"), base_url=self.api_base)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,  # Adjust based on qwen-plus context window
            num_output=512,
            model_name=self.model_name,
            is_chat_model=True
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> Any:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            extra_body={"enable_thinking": False}
        )
        return response.choices[0].message.content

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> Any:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            extra_body={"enable_thinking": False}
        )
        return response.choices[0].message.content

    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        # Streaming not implemented for simplicity; add if needed
        raise NotImplementedError("Streaming not supported in this example")

# 获取适用的embedding模型
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

# 1. Configure environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer parallelism warnings
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")  # Ensure API key is set

# 2. Load documents
data_dir = "./data"  # Replace with your document directory
if not os.path.exists(data_dir):
    print(f"❌ 数据目录 {data_dir} 不存在!")
    exit(1)
else:
    files = os.listdir(data_dir)
    print(f"✅ 数据目录 {data_dir} 存在, 包含 {len(files)} 个文件:")
    for file in files:
        print(f"  - {file} ({os.path.getsize(os.path.join(data_dir, file))} 字节)")

documents = SimpleDirectoryReader(data_dir).load_data()
print(f"✅ 成功加载 {len(documents)} 个文档")
for i, doc in enumerate(documents[:3]):  # 打印前3个文档的摘要
    print(f"  文档 {i+1}: 内容长度 {len(doc.text)} 字符, 开头: {doc.text[:50]}...")
if len(documents) > 3:
    print(f"  ... 还有 {len(documents)-3} 个文档")
# 4. 检查Milvus连接
if not check_milvus_connection():
    print("请确保Milvus服务正在运行")
    exit(1)
# 3. 获取合适的embedding模型
embed_model, embedding_dim = get_embedding_model()
if embed_model is None:
    print("❌ 无法配置任何embedding模型，退出程序")
    exit(1)

# 6. 配置Milvus向量存储
collection_name = "rag_collection"
print(f"检查集合 {collection_name} 是否已存在")
check_collection_exists(collection_name)
# 4. Configure Milvus vector store
vector_store = MilvusVectorStore(
    uri="http://localhost:19530",  # Milvus service address
    collection_name="rag_collection",  # Collection name
    dim=embedding_dim,  # 使用动态获取的向量维度
    overwrite=True,  # Overwrite collection if it exists (for development)
    index_params={"metric_type": "L2", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 64}},
    consistency_level="Strong",  # 确保强一致性
    primary_field="id",  # 使用id作为主键字段
    primary_field_type=pymilvus.DataType.VARCHAR,  # 指定主键字段类型为VARCHAR
    auto_id=False,  # 禁用自动ID生成，需要手动提供
    content_field="text"  # 指定内容字段
)
print("✅ 向量存储配置成功")
# 5. Create vector index
print("正在创建向量索引...")

# 在index构建前打印一些调试信息
print(f"将要处理 {len(documents)} 个文档...")
print(f"向量模型: {embed_model.__class__.__name__}")
print(f"向量维度: {embedding_dim}")
print(f"向量存储: {vector_store.__class__.__name__}")

# 测试向量生成
try:
    print("测试向量生成...")
    sample_text = documents[0].text[:100]
    print(f"样本文本: {sample_text}")
    sample_vector = embed_model.get_text_embedding(sample_text)
    print(f"样本向量维度: {len(sample_vector)}")
    print("✅ 向量生成测试成功")
except Exception as e:
    print(f"❌ 向量生成测试失败: {e}")

index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
    embed_model=embed_model,
    show_progress=True  # 显示进度
)
print("✅ 向量索引创建成功")
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

# 6. Configure Qianwen API using custom LLM
llm = QianwenLLM()

# 7. Create query engine
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3  # Retrieve top-3 similar documents
)

# 8. Execute RAG query
query = "LlamaIndex 支持哪些向量数据库？"  # Replace with your specific query
response = query_engine.query(query)
print(f"Query: {query}")
print(f"Response: {response}")