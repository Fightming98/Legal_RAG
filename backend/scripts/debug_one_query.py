"""单条管线调试脚本 — 在 VSCode 调试器下逐行跟完整 RAG 管线

用法:
    1. 确认仓库根目录 .env 已配置 SILICONFLOW_API_KEY（DeepSeek-V3.2 / BAAI/bge-m3）
    2. VSCode 中选中解释器 langchain1.2 环境
    3. 在 app/services/pipeline.py 各阶段函数里打断点
    4. 运行配置 "RAG: 调试单条管线 (直连 pipeline)" (F5)

    - 本脚本不走 HTTP / FastAPI, 直接构造 PipelineConfig 调 RAGPipeline.execute(),
      和 run_integration_test.py 是同一种跑法, 迭代更快。
    - 想在 API 层一起跟, 换用 "RAG: 启动后端" 配置 + 用 curl/browser 触发请求。
"""

import asyncio
import os
import sys

# 把 backend 目录加入 sys.path, 使脚本能 import app 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.pipeline import (
    RAGPipeline,
    PipelineConfig,
    QueryTransformStrategy,
    RerankStrategy,
    GenerationStrategy,
)

# ==================== 在这里改要跑的配置 ====================
QUESTION = "故意杀人罪的量刑标准是什么？"

CONFIG = PipelineConfig(
    query_transform=QueryTransformStrategy.HYDE,     # none / multi_query / hyde / decompose / multi_query_hyde
    rerank_strategy=RerankStrategy.BGE,              # none / simple / llm / bge
    generation_strategy=GenerationStrategy.STANDARD, # standard / chain_of_thought / self_reflect / structured_legal
    use_kg=True,                                     # 是否启用犯罪知识图谱
    top_k=5,
    collection_names=["laws", "cases"],
)
# ============================================================


async def main():
    # 首次调用会发起一次远端 API 请求，可在此验证硅基流动密钥/模型可用
    from app.core.llm import get_llm
    from app.core.embeddings import get_embeddings

    print("验证模型连通性...")
    await get_llm().ainvoke("你好")
    get_embeddings().embed_query("测试")
    print("连通性验证完成\n")

    pipe = RAGPipeline(CONFIG)
    resp = await pipe.execute(QUESTION)   # <-- 在这里打断点, F11 进入 execute()

    print("\n======== 结果 ========")
    print("回答:", resp.answer[:200])
    print("来源数:", len(resp.sources))
    print("KG 实体:", resp.kg_entities)
    print("管线配置:", resp.pipeline_config)
    print("指标:", resp.metrics)


if __name__ == "__main__":
    asyncio.run(main())
