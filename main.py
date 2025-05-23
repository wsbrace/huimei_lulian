import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import MilvusVectorStore
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. 配置环境变量（可选）
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免 tokenizer 并发警告

# 2. 加载文档
data_dir = "./data"  # 替换为你的文档目录
documents = SimpleDirectoryReader(data_dir).load_data()

# 3. 配置嵌入模型
# 使用 Hugging Face 的嵌入模型（例如 BAAI/bge-small-zh-v1.5，适合中文）
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 4. 配置 Milvus 向量存储
vector_store = MilvusVectorStore(
    uri="http://localhost:19530",  # Milvus 服务地址
    collection_name="rag_collection",  # 集合名称
    dim=384,  # 嵌入维度，需与嵌入模型匹配（bge-small-zh-v1.5 为 384）
    overwrite=True  # 如果集合已存在，覆盖它
)

# 5. 创建向量索引
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
    embed_model=embed_model
)

# 6. 配置 Qwen3 模型
model_name = "Qwen/Qwen-7B"  # 替换为你的 Qwen3 模型名称或本地路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16  # 使用半精度以节省显存
)

# 将 Qwen3 集成到 LlamaIndex
llm = HuggingFaceLLM(
    model_name=model_name,
    tokenizer=tokenizer,
    model=model,
    max_new_tokens=512,  # 最大生成 token 数
    generate_kwargs={"temperature": 0.7, "do_sample": True}
)

# 7. 创建查询引擎
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3  # 返回 top-3 相似文档
)

# 8. 执行 RAG 查询
query = "你的查询问题"  # 替换为你的具体问题
response = query_engine.query(query)
print(f"Query: {query}")
print(f"Response: {response}")