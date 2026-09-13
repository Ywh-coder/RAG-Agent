# RAG Agent

基于 Deep Agents 的 RAG Agent，用于检索并回答 LangChain 文档相关问题。

## 功能

1. 并发抓取并索引 LangChain 官方文档
2. 用户提问 → MMR 向量检索（多样性排序）→ chunk 写入 Agent 文件系统
3. 主 Agent 将每个 chunk 委托给 chunk-analyst 子 Agent 分析
4. 综合子 Agent 摘要，生成带引用链接的答案

## 要求

- Python 3.10+
- `deepagents >= 0.5.2`（StateBackend 支持 upload_files）
- 必须配置 checkpointer，否则子 Agent 读不到父 Agent 写入的文件
- 首次运行需下载 HuggingFace 模型（BAAI/bge-m3，约 2GB）

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
copy .env.example .env
```

| 变量名 | 用途 | 必填 |
|--------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek 对话模型 | 是 |
| `RETRIEVAL_K` | MMR 返回结果数 | 否（默认 6） |
| `FETCH_K` | MMR 预检索候选数 | 否（默认 20） |

> 模型自动从 HuggingFace 镜像站（hf-mirror.com）下载，国内无需科学上网。

## 使用

```bash
# 运行默认查询
python agent.py

# 自定义问题
python agent.py "How to use tools in LangChain?"

# 强制重新构建向量库
python agent.py --force-rebuild

# 检查向量库索引状态（不启动 Agent）
python agent.py --check-index
```

## 架构

```
用户问题
    ↓
max_marginal_relevance_search（MMR 检索 + 多样性去重）
    ↓
chunk 文件写入 /retrieved/{batch_id}/
    ↓
chunk-analyst 子 Agent（并行分析每个 chunk，max_tokens=1024）
    ↓
主 Agent 综合摘要，生成带引用答案
```

## 注意事项

- `chroma_db/` 目录在首次运行后自动生成，后续查询会复用已有索引
- 如需重建索引，使用 `--force-rebuild` 参数
- 检索策略已从相似度阈值过滤改为 MMR（Maximal Marginal Relevance），
  自动兼顾相关性得分和结果多样性，无需手动调 threshold
