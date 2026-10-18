# Legal RAG — 基于 LangChain 的轻量化法律智能问答系统

> 面向中国法律领域的检索增强生成（RAG）系统：以 LangChain 为编排框架，通过**硅基流动（SiliconFlow）** 云端 API 调用 **DeepSeek-V3.2** 生成模型与 **BGE-M3** 嵌入模型，结合 **ChromaDB** 向量数据库，实现了从文档加载、智能分块、向量化存储到语义检索与生成的完整 RAG 流程。

<p align="center">
  <strong>LangChain · DeepSeek-V3.2 · BGE-M3 · ChromaDB · FastAPI · React</strong>
</p>

---

## 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [可插拔 RAG 管线](#可插拔-rag-管线)
- [法律领域 RAG 优化方案](#法律领域-rag-优化方案)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [API 接口文档](#api-接口文档)
- [配置参数](#配置参数)
- [数据源与导入](#数据源与导入)
- [测试](#测试)
- [性能评估](#性能评估)
- [许可证](#许可证)

---

## 项目概述

本项目构建了一个**前后端分离**的轻量化 RAG 系统，并选择**中国法律领域**作为垂直应用场景进行验证。法律文本具有高度结构化（编/章/节/条/款/项）、术语精确、条文间存在引用关系、用户查询常为口语化表述等显著特征，对 RAG 系统的分块策略、检索精度和生成忠实度提出了严格要求。

系统针对这些领域特征实现了 **6 项专属 RAG 优化方案**（结构化分块 / 上下文标头 / HyDE / 术语规范化 / 犯罪知识图谱 / 自我反思），并提供了一个**四阶段可插拔的 RAG 管线**，各阶段策略可在前端可视化切换，方便做消融实验对比不同组合的效果。

> 该设计同样为将系统适配到其他垂直领域（医疗、金融、教育等）提供了可参考的方法论。

---

## 核心特性

- **云端模型服务**：LLM（DeepSeek-V3.2）与 Embedding（BGE-M3）均通过硅基流动 OpenAI 兼容 API 调用，本地零 GPU 依赖、响应快
- **法律领域深度优化**：6 项针对法律文本特征的专属 RAG 优化方案
- **可插拔管线架构**：查询变换（5 种）/ 重排序（3 种）/ 生成（4 种）阶段均支持策略切换，前端可视化配置
- **混合检索**：BM25（jieba 分词）稀疏检索 + BGE-M3 稠密向量检索 + RRF 融合，兼顾关键词精确匹配与语义召回
- **知识图谱增强**：内置 455+ 罪名犯罪知识图谱（定义 / 构成要件 / 量刑标准 / 相关法条），精确匹配并入检索结果
- **前后端分离**：FastAPI 异步后端 + React（Vite）单页前端
- **性能与质量评估**：内置 CPU / 内存实时监控、各阶段延迟分解、ROUGE / 忠实度 / 相关性自动评估
- **多源数据整合**：整合 6 个开源法律数据集，导入约 59,000 个文本分块

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                     Frontend (React + Vite)                        │
│   ┌───────────┐   ┌────────────┐   ┌──────────────┐              │
│   │ 智能问答    │   │ 知识库管理   │   │ 性能监控       │              │
│   │ 策略配置    │   │ 上传 / 重建  │   │ 基准测试 / 报告 │              │
│   └─────┬─────┘   └─────┬──────┘   └──────┬───────┘              │
│         └────────────────┼────────────────┘                       │
│                          │ HTTP / REST API                        │
├──────────────────────────┼────────────────────────────────────────┤
│                   Backend (FastAPI)                               │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                Pluggable RAG Pipeline                       │  │
│  │                                                            │  │
│  │  Stage 1 查询变换                                           │  │
│  │   [无 | 多查询扩展 | HyDE | 子问题分解 | 多查询+HyDE]        │  │
│  │              ↓                                             │  │
│  │  Stage 2 混合检索 (BM25 + 向量 + RRF)                       │  │
│  │   + Stage 2.5 犯罪知识图谱 (可选, 并行)                     │  │
│  │              ↓                                             │  │
│  │  Stage 3 重排序                                             │  │
│  │   [无 | 简单重排 (Jaccard+元数据) | LLM 重排]               │  │
│  │              ↓                                             │  │
│  │  Stage 4 生成                                               │  │
│  │   [标准 | 链式推理 CoT | 自我反思修正 | 结构化法律回答]       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────┐  ┌─────────────┐  ┌──────────┐  ┌─────────────┐ │
│  │  ChromaDB  │  │DeepSeek-V3.2│  │ BGE-M3   │  │ 犯罪知识图谱  │ │
│  │  向量数据库 │  │ (硅基流动)   │  │(硅基流动) │  │ (内存缓存)   │ │
│  └────────────┘  └─────────────┘  └──────────┘  └─────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 可插拔 RAG 管线

核心是 `app/services/pipeline.py` 中的 `RAGPipeline`，将 RAG 流程拆分为 4 个可独立配置的阶段：

```
用户提问 → PipelineConfig → RAGPipeline.execute()

  Stage 1 查询变换 (query_transform)
    ├── none             — 不变换，直接使用原始问题
    ├── multi_query      — LLM 从 3 个角度改写查询（法律条文 / 司法解释 / 不同术语）
    ├── hyde             — 生成假设性法律条文用于向量检索（原始问题仍用于 BM25）
    ├── decompose        — 将复杂问题分解为多个法律子问题，分别检索后合并
    └── multi_query_hyde — 多查询 + HyDE 组合（两个 LLM 调用并行执行）

  Stage 2 混合检索
    BM25 (jieba 分词) + BGE-M3 向量检索 → RRF 融合
    + Stage 2.5 犯罪知识图谱查找（可选，并行，精确匹配罪名）

  Stage 3 重排序 (rerank_strategy)
    ├── none   — 不重排，直接使用 RRF 融合的原始排序
    ├── simple — Jaccard 关键词重叠 + 法律名称 / 条号 / 案号元数据加分
    └── llm    — 简单重排预筛选后，LLM 单次批量评分

  Stage 4 生成 (generation_strategy)
    ├── standard         — 标准法律问答 Prompt
    ├── chain_of_thought — 链式推理（识别问题 → 查找法律 → 分析要件 → 结论 → 注意事项）
    ├── self_reflect     — 标准生成 + 自我反思验证 + 必要时修正重生成
    └── structured_legal — 结构化输出（法律结论 / 适用法律 / 详细分析 / 注意事项）
```

**向后兼容**：旧版 API 参数自动映射 `use_rerank=false → rerank_strategy="none"`、`use_query_rewrite=true → query_transform="multi_query"`。

---

## 法律领域 RAG 优化方案

法律文本与通用文本有显著差异，主流 RAG 方案在面对这些特征时存在明确短板，本项目针对每一项短板给出了专属解决方案。

| # | 优化方案 | 解决的领域特有问题 | 实现文件 |
|---|----------|--------------------|----------|
| 1 | **法律条文结构化分块** | 固定窗口分块会把一条完整法条切成两半，语义破碎；以「条」为原子单位按编/章/节/条/款/项层级切分，案例按「裁判要旨/案情/理由/结果」结构切分 | `utils/legal_chunker.py` |
| 2 | **上下文标头注入** | 裸文本脱离「属于哪部法律、第几条」的上下文，Embedding 无法捕捉其法律归属；注入 `[法律名称 \| 章节 \| 条号]` 结构化标头，**零 LLM 成本**（区别于 Anthropic Contextual Retrieval 的 LLM 生成式方案） | `utils/legal_chunker.py` |
| 3 | **HyDE 假设文档检索** | 用户「问题」与法律「条文」的 Embedding 分布差异大（query-document mismatch）；LLM 先生成假设性法条再做向量检索，且**分离式检索**（假设文档仅用于向量检索，原始问题仍用于 BM25 保留精确匹配） | `services/hyde.py` + `core/retriever.py` |
| 4 | **法律术语规范化 + 查询分解** | 口语与法律术语鸿沟巨大（「打人」→「故意伤害罪」、「老赖」→「失信被执行人」）；内置 21 组口语→术语映射表 + LLM 复杂问题子问题分解 | `services/query_rewriter.py` |
| 5 | **犯罪知识图谱增强 (KAG)** | 纯向量检索可能返回语义接近但领域错误的条文；用内存字典实现轻量 KG，精确匹配罪名并返回定义 / 四要件 / 量刑 / 法条，合并到检索结果前列 | `services/kg_service.py` |
| 6 | **自我反思修正** | 防止 LLM 幻觉不存在的法条编号（如编造「民法典第999条」）；生成后验证引用是否存在于参考资料中，最多 1 轮修正（增加约 3-5s）在准确性与速度间取平衡 | `services/self_reflect.py` |

---

## 技术栈

| 组件 | 选择 | 说明 |
|------|------|------|
| 主模型 | DeepSeek-V3.2（硅基流动 API） | 云端强模型，无本地显存负担 |
| Embedding | BGE-M3（硅基流动 API） | 多语言多粒度，中文检索效果好 |
| 向量数据库 | ChromaDB | 轻量嵌入式、零配置部署、自带持久化与元数据过滤 |
| 稀疏检索 | BM25（rank-bm25 + jieba） | 精确匹配法条编号与关键词 |
| RAG 框架 | LangChain | 组件化设计、生态成熟 |
| 后端 | FastAPI | 异步高性能、自动生成 OpenAPI 文档 |
| 前端 | React + Vite | 开发体验好、HMR 快速 |

---

## 项目结构

```
Legal_rag/
├── README.md                          # 本文件
├── .gitignore
│
├── backend/
│   ├── requirements.txt               # Python 依赖
│   ├── .env.example                   # 环境变量示例（复制为 .env 使用）
│   ├── app/
│   │   ├── main.py                    # FastAPI 应用入口（含 /health 健康检查）
│   │   ├── config.py                  # 全局配置（pydantic-settings，读取 .env）
│   │   ├── api/                       # HTTP API 路由层
│   │   │   ├── chat.py                # /api/chat — 问答（含流式 SSE）、记录
│   │   │   ├── knowledge.py           # /api/knowledge/* — 知识库管理
│   │   │   ├── performance.py         # /api/performance/* — 性能监控与报告
│   │   │   └── settings.py            # /api/settings/* — 密钥策略与 Key 校验
│   │   ├── core/                      # 核心组件
│   │   │   ├── api_key.py             # 请求级用户 API Key 上下文（ContextVar）
│   │   │   ├── llm.py                 # DeepSeek-V3.2 初始化（ChatOpenAI → 硅基流动）
│   │   │   ├── embeddings.py          # BGE-M3 初始化（按请求动态解析 Key 的代理）
│   │   │   ├── vectorstore.py         # ChromaDB 向量数据库（带实例缓存）
│   │   │   ├── retriever.py           # 混合检索器（BM25 + 向量 + RRF）
│   │   │   └── logging.py             # 日志配置
│   │   ├── services/                  # 业务逻辑层
│   │   │   ├── pipeline.py            # 可插拔 RAG 管线编排器（策略枚举 + PipelineConfig）
│   │   │   ├── prompts.py             # 全部 Prompt 模板
│   │   │   ├── hyde.py                # HyDE 假设文档生成
│   │   │   ├── self_reflect.py        # 自我反思与修正
│   │   │   ├── kg_service.py          # 犯罪知识图谱加载、实体提取与查找
│   │   │   ├── rag_service.py         # RAG 入口（向后兼容包装器）
│   │   │   ├── stream_service.py      # 流式 RAG 事件流（思考/回答/来源/指标）
│   │   │   ├── kb_service.py          # 知识库管理（上传/列表/删除/重建）
│   │   │   ├── reranker.py            # 重排序（Jaccard + 元数据 / LLM 批量评分）
│   │   │   ├── query_rewriter.py      # 查询重写 + 术语规范化 + 查询分解
│   │   │   ├── perf_service.py        # 性能基准测试与系统监控
│   │   │   ├── quality_service.py     # 质量评估（ROUGE / 相关性 / 忠实度）
│   │   │   └── report_service.py      # 性能报告与问答记录生成
│   │   ├── models/
│   │   │   └── schemas.py             # Pydantic 数据模型
│   │   └── utils/
│   │       ├── legal_chunker.py       # 法律文本专用分块器 + 上下文标头注入
│   │       └── metadata.py            # 元数据格式化工具
│   ├── data/                          # 数据目录（已被 .gitignore 排除，不随仓库上传）
│   │   ├── laws/                      # 法律条文 + 犯罪知识图谱（原始数据）
│   │   ├── laws_slight/               # 法条精简版（导入向量库的数据源）
│   │   ├── cases/                     # 案例数据
│   │   ├── cases_slight/              # 案例精简版（导入向量库的数据源）
│   │   ├── qa/                        # 法律问答语料
│   │   ├── reference/                 # 评估参考数据
│   │   ├── reports/                   # 生成的性能报告
│   │   └── chat_records/              # 问答性能记录
│   ├── scripts/                       # 数据处理与工具脚本
│   │   ├── prepare_datasets.py        # 数据集格式转换与下载
│   │   ├── import_data.py             # 数据分块并写入 ChromaDB
│   │   ├── convert_hf_laws.py         # HuggingFace 法律法规格式转换
│   │   ├── crawl_laws.py              # 法律法规数据库爬虫（备用）
│   │   ├── cleanup_chroma.py          # ChromaDB 清理工具
│   │   └── run_quality_eval.py        # 批量质量评估
│   └── tests/                         # 单元测试
│       ├── test_basic.py              # 配置加载、分块器
│       ├── test_pipeline.py           # 管线配置与策略枚举
│       ├── test_advanced_retrieval.py # 术语规范化、上下文标头
│       ├── test_kg.py                 # 犯罪知识图谱模块
│       ├── test_api_key.py            # 用户自带 Key：校验、上下文、401 行为
│       └── test_queries.txt           # 测试问题集
│
└── frontend/
    ├── package.json                   # 依赖与脚本
    ├── vite.config.js                 # Vite 配置 + /api 代理到后端 8000 端口
    ├── index.html                     # HTML 入口
    └── src/
        ├── main.jsx                   # React 挂载入口
        ├── App.jsx                    # 根组件（侧边栏导航 + 页面切换 + Key 设置入口）
        ├── components/
        │   └── ApiKeySettings.jsx     # 用户 API Key 设置弹窗（校验 + 本地保存）
        ├── pages/
        │   ├── ChatPage.jsx           # 智能问答页面（管线策略配置面板）
        │   ├── KnowledgePage.jsx      # 知识库管理页面
        │   └── PerformancePage.jsx    # 性能监控页面
        ├── services/
        │   └── api.js                 # Axios API 封装
        └── styles/
            └── global.css             # 全局样式
```

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/<your-username>/Legal_rag.git
cd Legal_rag
```

### 2. 配置硅基流动 API

本项目模型由硅基流动（SiliconFlow）云端 API 提供，**无需本地部署模型**。支持两种密钥模式：

- **用户自带 Key（默认，推荐公网部署）**：每个访客在页面左下角「API 设置」中填入自己的 Key，密钥只保存在其浏览器 `localStorage`，随请求通过 `X-SiliconFlow-API-Key` 请求头传给后端。未配置的访客无法使用问答、上传、重建索引和性能测试。服务端 `backend/.env` 中的 Key 仅为脚本和健康检查兜底，**不会被普通请求使用**。
- **服务端统一 Key（本地开发/内网）**：设置 `REQUIRE_USER_API_KEY=false`，未携带 Key 的请求回退使用服务端 `.env` 中的 Key。

本地开发可直接使用服务端 Key（跳过页面配置）：

```bash
cd backend
cp .env.example .env
# 编辑 .env
```

```bash
SILICONFLOW_API_KEY=sk-xxxxx
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
REQUIRE_USER_API_KEY=false     # 本地开发回退到服务端 Key；公网部署请保持 true
```

公网部署时建议将 `SILICONFLOW_API_KEY` 留空，并保持 `REQUIRE_USER_API_KEY=true`，从源头上避免访客消耗部署者的额度。

使用的默认模型：
- 生成：`deepseek-ai/DeepSeek-V3.2`
- Embedding：`BAAI/bge-m3`

> **注意**：`config.py` 按进程工作目录读取 `.env`，因此在 `backend/` 目录下启动时，`.env` 需放在 `backend/` 内。

### 3. 启动后端

```bash
conda create -n legal_rag python=3.10 -y
conda activate legal_rag
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

验证服务：访问 `http://localhost:8000/health`，当 `provider.status` 为 `running` 时表示硅基流动密钥与模型连通正常。API 文档见 `http://localhost:8000/docs`。

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173`（Vite 开发服务器已将 `/api/*` 代理到 `http://localhost:8000`）。

### 5. 准备数据与导入（首次使用）

> 原始数据集未包含在仓库中（体积大，且受各数据集自身许可约束）。请按下方[数据源与导入](#数据源与导入)章节获取数据并放置到 `backend/data/` 后，再执行导入：

```bash
conda activate legal_rag
cd backend
python -m scripts.prepare_datasets   # 格式转换：ZIP → JSONL、CrimeKG → TXT、QA → 参考答案
python -m scripts.import_data        # 分块并写入 ChromaDB 向量数据库
```

也可通过前端「知识库管理」页面上传文档，或调用 `POST /api/knowledge/rebuild` 重建索引。

> **注意**：切换 Embedding 模型后，旧的 `backend/chroma_db` 向量库不再兼容，**必须重建索引**。

### 6. 运行测试

```bash
conda activate legal_rag
cd backend
pytest tests/ -v
```

---

## API 接口文档

所有接口响应统一包装为 `{ "code": 200, "data": ..., "message": "success" }`。启动后端后亦可直接在 `http://localhost:8000/docs`（Swagger UI）交互调试。

> **鉴权说明**：默认配置下，所有消耗模型的接口（`/api/chat`、`/api/knowledge/upload`、`/api/knowledge/rebuild`、`/api/performance/bench`、`/api/performance/report`）都必须携带用户自己的 Key：
>
> ```
> X-SiliconFlow-API-Key: sk-xxxxx
> ```
>
> 未携带返回 `401`，Key 无效返回 `401`，额度不足/限流返回 `429`，无法连接硅基流动返回 `502`。只读列表、下载、健康检查接口不需要 Key。

### 密钥设置

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/settings/key-policy` | 公开的密钥策略（是否强制用户自带 Key、请求头名称等，不含密钥内容） |
| POST | `/api/settings/verify-key` | 用请求头中的 Key 调用硅基流动 `/models` 校验，返回脱敏 Key 与可用模型数 |

### 健康检查

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 服务健康状态、模型配置、硅基流动 provider 状态、知识库统计（`provider.status` 可能为 `not_configured`：服务端未配置兜底 Key） |

### 问答

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | RAG 智能问答（一次性返回完整结果） |
| POST | `/api/chat/stream` | RAG 流式问答（SSE：思考过程 + 回答正文增量） |
| POST | `/api/chat/save-record` | 保存一次问答性能记录 |
| GET | `/api/chat/records` | 列出问答性能记录 |
| GET | `/api/chat/records/{record_id}` | 下载问答性能记录 JSON |

**POST `/api/chat` 请求参数**：

```json
{
  "question": "故意杀人罪的量刑标准是什么？",
  "top_k": 5,
  "collection": "all",
  "query_transform": "hyde",
  "rerank_strategy": "simple",
  "generation_strategy": "chain_of_thought",
  "use_kg": true,
  "evaluate_quality": false,
  "monitor_system": false
}
```

| 参数 | 可选值 | 默认值 | 说明 |
|------|--------|--------|------|
| `question` | — | 必填 | 用户问题（1-2000 字） |
| `query_transform` | `none` / `multi_query` / `hyde` / `decompose` / `multi_query_hyde` | `none` | 查询变换策略 |
| `rerank_strategy` | `none` / `simple` / `llm` | `simple` | 重排序策略 |
| `generation_strategy` | `standard` / `chain_of_thought` / `self_reflect` / `structured_legal` | `standard` | 生成策略 |
| `use_kg` | `true` / `false` | `false` | 犯罪知识图谱增强 |
| `collection` | `all` / `laws` / `cases` | `all` | 检索范围 |
| `top_k` | 1-20 | `5` | 最终返回文档数 |
| `evaluate_quality` | `true` / `false` | `false` | 是否评估回答质量 |
| `monitor_system` | `true` / `false` | `false` | 是否采集系统资源快照 |
| `enable_thinking` | `true` / `false` | `true` | 是否开启 DeepSeek 思考模式（开启后流式接口返回 `reasoning_content`，思考过程可在前端折叠展开） |

**响应（节选）**：

```json
{
  "answer": "根据《中华人民共和国刑法》第二百三十二条...",
  "sources": [ { "content": "...", "metadata": {...}, "score": 0.86 } ],
  "metrics": {
    "query_rewrite_ms": 1500.0,
    "retrieval_ms": 320.5,
    "kg_lookup_ms": 2100.0,
    "rerank_ms": 15.2,
    "generation_ms": 3500.8,
    "self_reflect_ms": null,
    "total_ms": 7450.3,
    "was_corrected": false
  },
  "kg_entities": ["故意杀人罪"],
  "generation_strategy": "chain_of_thought",
  "pipeline_config": { ... }
}
```

**POST `/api/chat/stream`（SSE）**：

请求体与 `/api/chat` 相同，响应为 `text/event-stream`，前端逐事件更新界面：

| 事件 | 数据 | 说明 |
|------|------|------|
| `stage` | `{ "stage": "...", "message": "..." }` | 管线阶段提示（检索 / 生成 / 反思 / 质量评估） |
| `sources` | `{ "sources": [...], "rewritten_queries": ..., "kg_entities": ..., "metrics": ... }` | 检索完成，返回来源与检索耗时 |
| `reasoning` | `{ "delta": "..." }` | 模型思考内容增量（siliconflow `reasoning_content`） |
| `answer` | `{ "delta": "..." }` | 回答正文增量 |
| `answer_replace` | `{ "answer": "...", "was_corrected": true }` | 自我反思修正后的完整回答（替换此前流式内容） |
| `metrics` | `{ ... }` | 最终阶段耗时 |
| `quality` | `{ ... }` | 质量评估结果（开启时） |
| `done` | `{ "generation_strategy": ..., "system_before": ..., "system_after": ... }` | 流结束元数据 |
| `error` | `{ "detail": "...", "status": 401 }` | 流内异常（开始推流后 HTTP 状态码无法再修改） |

前端「深度思考」开关控制是否开启思考模式；思考内容默认折叠，点击回答上方的「思考过程」标题手动展开。

### 知识库管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/knowledge/upload` | 上传文档（支持 `.txt` `.md` `.json`） |
| GET | `/api/knowledge/list` | 列出已上传文件 |
| DELETE | `/api/knowledge/{filename}` | 删除指定文件 |
| POST | `/api/knowledge/rebuild` | 重建全部索引 |
| GET | `/api/knowledge/stats` | 知识库统计（文件数、分块数） |

### 性能监控

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/performance/system` | CPU / 内存实时状态 |
| POST | `/api/performance/bench` | 运行基准测试 |
| POST | `/api/performance/report` | 生成性能测试报告 |
| GET | `/api/performance/reports` | 列出性能报告 |
| GET | `/api/performance/reports/{report_id}` | 获取完整性能报告 |

---

## 配置参数

通过环境变量或 `.env` 文件覆盖默认配置（完整定义见 `app/config.py`）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `SILICONFLOW_API_KEY` | 空 | 服务端兜底密钥（用户自带 Key 模式下仅脚本/健康检查使用，公网部署建议留空） |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1` | 硅基流动 OpenAI 兼容端点 |
| `REQUIRE_USER_API_KEY` | `true` | 是否强制请求携带用户自己的 Key（`false` 时回退服务端 Key） |
| `API_KEY_HEADER` | `X-SiliconFlow-API-Key` | 前端携带用户 Key 的请求头名称 |
| `LLM_MODEL` | `deepseek-ai/DeepSeek-V3.2` | 生成模型 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding 模型 |
| `LLM_TEMPERATURE` | `0.3` | 生成温度 |
| `RETRIEVAL_TOP_K` | `10` | 初始检索候选条数 |
| `BM25_WEIGHT` / `VECTOR_WEIGHT` | `0.5` / `0.5` | BM25 / 向量检索融合权重 |
| `RERANK_TOP_K` | `5` | 重排序后最终保留条数 |
| `CONTEXTUAL_CHUNKING` | `true` | 是否启用上下文标头注入 |
| `HYDE_TEMPERATURE` | `0.7` | HyDE 假设文档生成温度 |
| `SELF_REFLECT_MAX_ITER` | `1` | 自我反思最大修正轮次 |
| `DEFAULT_QUERY_TRANSFORM` | `none` | 默认查询变换策略 |
| `DEFAULT_RERANK` | `simple` | 默认重排序策略 |
| `DEFAULT_GENERATION` | `standard` | 默认生成策略 |

---

## 数据源与导入

### 数据来源

| 数据集 | 来源 | 用途 |
|--------|------|------|
| **Chinese-Laws** | [ModelScope dengcao/Chinese-Laws](https://www.modelscope.cn/datasets/dengcao/Chinese-Laws) | 177 部法律全文（TXT） |
| **HF 法律法规库** | [HuggingFace twang2218/chinese-law-and-regulations](https://huggingface.co/datasets/twang2218/chinese-law-and-regulations) | 2,719 部法规全文 |
| **犯罪知识图谱** | [GitHub liuhuanyong/CrimeKgAssitant](https://github.com/liuhuanyong/CrimeKgAssitant) | 455+ 罪名定义 / 构成要件 / 量刑 |
| **CAIL2018** | [CAIL 官方](https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip) | 刑事法律文书 |
| **CAIL2019-SCM** | [GitHub thunlp/CAIL](https://github.com/thunlp/CAIL) | 民事案例相似性匹配 |
| **CrimeKgAssitant QA** | 同上 | 20 万+ 条真实法律咨询问答对 |

### 分块与存储

经预处理与导入后，文本分块存储于 ChromaDB 的两个 Collection：

| Collection | 数据来源 | 分块策略 | 分块参数 |
|-----------|---------|---------|---------|
| `laws` | Chinese-Laws + HF 法律法规库 + CrimeKG | `LegalArticleSplitter`：按编/章/节/条层级切分 | chunk_size=512, overlap=64 |
| `cases` | CAIL2019-SCM + CAIL2018 | `LegalCaseSplitter`：按裁判要旨/案情/理由/结果切分 | chunk_size=1024, overlap=128 |

导入脚本：

```bash
python -m scripts.prepare_datasets   # 格式转换与下载
python -m scripts.import_data        # 分块并写入 ChromaDB
```

---

## 测试

系统包含 24 个单元测试用例，分布在 4 个测试文件中：

| 测试文件 | 覆盖范围 |
|---------|---------|
| `test_basic.py` | 配置加载、法律条文分块、案例分块 |
| `test_pipeline.py` | PipelineConfig 默认值、策略枚举、向后兼容映射 |
| `test_advanced_retrieval.py` | 法律术语规范化、上下文标头注入 |
| `test_kg.py` | 知识图谱加载、罪名查找 |

```bash
cd backend && pytest tests/ -v
```

---

## 性能评估

系统内置性能测试模块，覆盖以下维度：

- **资源占用**：CPU / RAM 实时监控（前后快照对比）
- **响应速度**：端到端延迟、各阶段耗时分解（查询变换 / 检索 / KG / 重排序 / 生成 / 反思）
- **生成质量**：ROUGE、检索相关性、忠实度自动评估
- **吞吐量**：并发请求处理能力（基准测试 QPS）

**各策略组合预期延迟**（受模型响应速度影响）：

| 策略组合 | 查询变换 | 重排序 | 生成 | KG | 预期延迟 |
|---------|---------|--------|------|-----|---------|
| 快速 | 无 | 简单 | 标准 | 关 | 10-20s |
| 精确检索 | HyDE | 简单 | 标准 | 关 | 15-25s |
| 深度分析 | HyDE | 简单 | CoT | 开 | 25-40s |
| 最高质量 | 多查询+HyDE | LLM | 自我修正 | 开 | 40-80s |

---

## 许可证

本项目仅供学习与研究使用，法律数据版权归原始数据提供方所有。
