import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

# 1. Configure environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer parallelism warnings
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")  # Ensure API key is set

# 2. Load documents
data_dir = "./data"  # Replace with your document directory
documents = SimpleDirectoryReader(data_dir).load_data()

# 3. Configure embedding model
# Using BAAI/bge-small-zh-v1.5 for Chinese documents
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 4. Configure Milvus vector store
vector_store = MilvusVectorStore(
    uri="http://localhost:19530",  # Milvus service address
    collection_name="rag_collection",  # Collection name
    dim=384,  # Embedding dimension, must match the embedding model (bge-small-zh-v1.5 is 384)
    overwrite=True  # Overwrite collection if it exists (for development)
)

# 5. Create vector index
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
    embed_model=embed_model
)

# 6. Configure Qianwen API using OpenAI-compatible client
llm = OpenAI(
    model="qwen-plus",  # Choose model: qwen-turbo, qwen-plus, or qwen-max
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    max_tokens=512,  # Maximum generated tokens
    temperature=0.7,
    top_p=0.9,
    additional_kwargs={"enable_thinking": False}  # Required for Qwen commercial API
)

# 7. Create query engine
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3  # Retrieve top-3 similar documents
)

# 8. Execute RAG query
query = "你的查询问题"  # Replace with your specific query
response = query_engine.query(query)
print(f"Query: {query}")
print(f"Response: {response}")