"""全局配置"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings

# 本仓库根目录（含共享 .env：SILICONFLOW / DEEPSEEK 等平台密钥）。
# config.py 位于 backend/app/ 下，向上 4 级即 langchain1.2_tutorial。
# 注意：pydantic-settings 的 env_file 默认按进程 CWD 解析相对路径，
# 而本应用既可能从 backend/ 启动也可能从仓库根启动，因此显式把根目录
# .env 的绝对路径也列入 env_file，保证两种启动方式都能读到平台密钥。
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# 重要：本机系统代理（如 Clash/Verge，注册表 ProxyEnable=1）会拦截对 localhost
# 的请求并返回 502。Python 的 httpx/requests 只认 NO_PROXY 环境变量，不读取
# Windows 注册表里的 ProxyOverride 绕过列表。因此在创建任何 HTTP 客户端之前，
# 先把 localhost / 127.0.0.1 加入 NO_PROXY（本地其他服务仍可能走 localhost）。
for _key in ("NO_PROXY", "no_proxy"):
    _hosts = [h.strip() for h in os.environ.get(_key, "").split(",") if h.strip()]
    for _host in ("localhost", "127.0.0.1"):
        if _host not in _hosts:
            _hosts.append(_host)
    os.environ[_key] = ",".join(_hosts)


class Settings(BaseSettings):
    # --- 应用 ---
    APP_NAME: str = "法律 RAG 系统"  # 应用名称
    APP_VERSION: str = "1.0.0"  # 应用版本号
    API_PREFIX: str = "/api"  # API 路由统一前缀
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]  # 允许跨域访问的前端来源

    # --- 用户自带 API Key ---
    # 为 True 时，所有消耗模型的接口都必须由请求头携带用户自己的 Key，
    # 服务端 .env 中的 SILICONFLOW_API_KEY 不再为普通请求兜底（仅脚本、
    # 健康检查等非请求场景使用），避免公网访客消耗部署者的额度。
    REQUIRE_USER_API_KEY: bool = True
    API_KEY_HEADER: str = "X-SiliconFlow-API-Key"  # 前端携带密钥的请求头名称

    # --- 硅基流动 (SiliconFlow) / 模型 ---
    # 从 .env 读取，见仓库根目录 .env 的 SILICONFLOW_API_KEY / SILICONFLOW_BASE_URL
    SILICONFLOW_API_KEY: str = ""  # 硅基流动 API 密钥
    SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"  # 硅基流动 OpenAI 兼容端点
    LLM_MODEL: str = "deepseek-ai/DeepSeek-V3.2"  # 大语言模型名称
    LLM_TEMPERATURE: float = 0.3  # 生成温度（越低越确定，越高越随机）
    LLM_TIMEOUT: int = 60  # 模型调用超时时间（秒）
    EMBEDDING_MODEL: str = "BAAI/bge-m3"  # 文本嵌入（向量化）模型

    # --- ChromaDB ---
    CHROMA_PERSIST_DIR: str = str(Path(__file__).resolve().parent.parent / "chroma_db")  # ChromaDB 持久化目录
    LAWS_COLLECTION: str = "laws"  # 法条向量集合名
    CASES_COLLECTION: str = "cases"  # 案例向量集合名

    # --- 检索 ---
    RETRIEVAL_TOP_K: int = 10  # 初始检索返回的候选条数
    BM25_WEIGHT: float = 0.5  # BM25（关键词）检索权重
    VECTOR_WEIGHT: float = 0.5  # 向量检索权重
    RERANK_TOP_K: int = 5  # 重排序后最终保留条数

    # --- 数据目录 ---
    DATA_DIR: str = str(Path(__file__).resolve().parent.parent / "data")  # 数据根目录
    LAWS_DIR: str = str(Path(__file__).resolve().parent.parent / "data" / "laws_slight")  # 法条数据目录
    CASES_DIR: str = str(Path(__file__).resolve().parent.parent / "data" / "cases_slight")  # 案例数据目录
    REPORTS_DIR: str = str(Path(__file__).resolve().parent.parent / "data" / "reports")  # 报告数据目录
    CHAT_RECORDS_DIR: str = str(Path(__file__).resolve().parent.parent / "data" / "chat_records")  # 对话记录目录

    # --- 高级 RAG 管线 ---
    CONTEXTUAL_CHUNKING: bool = True  # 是否启用上下文分块
    KG_DATA_PATH: str = str(  # 犯罪知识图谱数据路径
        Path(__file__).resolve().parent.parent / "data" / "laws" / "CrimeKG" / "犯罪知识图谱.txt"
    )
    HYDE_TEMPERATURE: float = 0.7  # HYDE（假设性文档嵌入）生成温度
    SELF_REFLECT_MAX_ITER: int = 1  # 自反思（self-reflect）最大迭代次数
    DEFAULT_QUERY_TRANSFORM: str = "none"  # 默认查询改写策略（none / hyde 等）
    DEFAULT_RERANK: str = "simple"  # 默认重排序策略（simple 等）
    DEFAULT_GENERATION: str = "standard"  # 默认生成策略（standard 等）

    # 注：本机 / 项目根目录可能有一个共享的 .env（含 DEEPSEEK、ZHIPUAI、LANGSMITH
    # 等与本项目无关的变量）。pydantic-settings>=2.12 默认 extra='forbid'，
    # 不忽略这些额外键会直接报 ValidationError，因此这里显式设为 ignore。
    # env_file 同时加载进程 CWD 的 .env（backend/.env，若存在）与仓库根目录
    # .env（绝对路径），保证两种启动方式都能读到 SILICONFLOW_API_KEY。
    model_config = {
        "env_file": (".env", str(PROJECT_ROOT / ".env")),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# 实例化全局配置，供全项目统一 import 引用（import_data 等脚本据此读取 LAWS_DIR/CASES_DIR 等路径）
settings = Settings()
