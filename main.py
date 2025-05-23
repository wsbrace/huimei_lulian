import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. Configure environment variables (optional)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer parallelism warnings

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

# 6. Configure Qwen3 model
model_name = "Qwen/Qwen-7B"  # Replace with your Qwen3 model name or local path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16  # Use half-precision to save memory
)

# Integrate Qwen3 with LlamaIndex
llm = HuggingFaceLLM(
    model_name=model_name,
    tokenizer=tokenizer,
    model=model,
    max_new_tokens=512,  # Maximum generated tokens
    generate_kwargs={"temperature": 0.7, "do_sample": True}
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