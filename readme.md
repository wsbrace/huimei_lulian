# LlamaIndex 与 通义千问 集成项目
pip3 install --upgrade llama-index-core llama-index-llms-openai llama-index-vector-stores-milvus llama-index-embeddings-huggingface openai 
pymilvus 要升级到最新
这个项目演示如何使用 LlamaIndex、Milvus 和通义千问大模型构建 RAG (检索增强生成) 应用程序。

## 环境要求

- Python 3.8+
- Milvus 向量数据库服务 (可通过 Docker 启动)
- 通义千问 API 密钥 (DASHSCOPE_API_KEY)

## 安装依赖

```bash
pip install llama-index
pip install llama-index-vector-stores-milvus
pip install llama-index-embeddings-huggingface
pip install llama-index-embeddings-dashscope
pip install pymilvus
pip install openai
pip install sentence-transformers
```

## 文件说明

- **qianwen.py**: 主程序，实现了通义千问与LlamaIndex的集成
- **check.py**: 检查和调试工具，用于验证文档导入是否成功
- **debug_milvus.py**: Milvus 数据库调试工具，用于查看集合中的数据情况
- **direct_import.py**: 直接使用 pymilvus API 导入数据的工具

## 使用方法

### 1. 启动 Milvus 服务

如果使用 Docker:
```bash
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest standalone
```

### 2. 准备数据

在项目根目录创建 `data` 文件夹，放入需要导入的文档。

### 3. 设置环境变量

```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
```

### 4. 运行主程序

```bash
python qianwen.py
```

### 5. 排障工具

如果遇到文档未成功导入 Milvus 的问题，可以尝试使用以下工具:

1. 检查导入问题:
```bash
python check.py
```

2. 查看 Milvus 数据:
```bash
python debug_milvus.py
```

3. 绕过 LlamaIndex 直接导入:
```bash
python direct_import.py
```

## 注意事项

1. Milvus 集合中存储文档时必须指定 `id` 字段
2. 默认使用阿里云文心 embedding 模型，也支持本地 Hugging Face 模型
3. 文档首次导入可能需要下载模型，请确保网络连接正常

## 问题排查

- 如果遇到 "集合中没有记录" 的问题，可能是因为:
  - Milvus 服务未正确启动
  - embedding 生成过程失败
  - 集合的字段定义不符合预期 (特别是 `id` 字段)

尝试使用 `direct_import.py` 脚本来验证文档能否正确导入。