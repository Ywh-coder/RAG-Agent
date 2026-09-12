# RAG Agent

基于 Deep Agents 的 RAG Agent，用于检索并回答 LangChain 文档相关问题。

## 功能

1. 并发抓取并索引 LangChain 官方文档
2. 用户提问 → 向量检索 → chunk 写入 Agent 文件系统
3. 主 Agent 将每个 chunk 委托给 chunk-analyst 子 Agent 分析
4. 综合子 Agent 摘要，生成带引用链接的答案

## 要求

- Python 3.10+
- `deepagents >= 0.5.2`（StateBackend 支持 upload_files）
- 必须配置 checkpointer，否则子 Agent 读不到父 Agent 写入的文件

## 安装

```bash
# 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 配置

复制 `.env.example` 并填入你的 API Key：

```bash
cp .env.example .env
```

需要以下环境变量：

| 变量名 | 用途 |
|--------|------|
| `OPENAI_API_KEY` | OpenAI 嵌入模型（text-embedding-3-large） |
| `DEEPSEEK_API_KEY` | DeepSeek 对话模型 |

## 使用

```bash
# 运行默认查询
python agent.py

# 自定义问题
python agent.py "How to use tools in LangChain?"

# 强制重新构建向量库
python agent.py --force-rebuild

# 调整相似度阈值
python agent.py --threshold 0.5
```

## 架构

```
用户问题
    ↓
search_documentation（向量检索 + 阈值过滤）
    ↓
chunk 文件写入 /retrieved/{batch_id}/
    ↓
chunk-analyst 子 Agent（并行分析每个 chunk）
    ↓
主 Agent 综合摘要，生成带引用答案
```

## 注意事项

- `chroma_db/` 目录在首次运行后自动生成，后续查询会复用已有索引
- 如需重建索引，使用 `--force-rebuild` 参数
