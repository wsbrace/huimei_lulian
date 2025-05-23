import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import OpenAI
from typing import Any, Optional
from pydantic import BaseModel, Field

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

# 6. Configure Qianwen API using custom LLM
llm = QianwenLLM()

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