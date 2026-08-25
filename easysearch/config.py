"""EasySearch 配置文件。

M1 安全基座：源码零密钥。所有敏感凭证从环境变量 / .env 文件读取，
本文件不再硬编码任何 API Key。请将密钥写入项目根目录的 .env 文件：

    DASHSCOPE_API_KEY=sk-你的Key
    DEEPSEEK_API_KEY=sk-你的Key

.env 已加入 .gitignore，不会提交；.env.example 提供模板。
"""

import os

# 软依赖 python-dotenv：安装后自动加载项目根 .env，未安装则回退纯环境变量。
try:  # pragma: no cover - 依赖可选
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _env_float(name: str, default: str) -> float:
    """从环境变量读取浮点配置，缺省返回 default。"""
    return float(os.getenv(name, default))


def _env_int(name: str, default: str) -> int:
    """从环境变量读取整数配置，缺省返回 default。"""
    return int(os.getenv(name, default))


def _env_bool(name: str, default: str) -> bool:
    """从环境变量读取布尔配置（"1"/"true" 视为 True，其余 False）。"""
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


# DashScope API Key（qwen3.7-text-embedding / qwen3-vl-rerank）
# 获取地址：https://dashscope.console.aliyun.com/apiKey
# M1：源码零密钥，从环境变量 / .env 读取。请勿在此硬编码。
DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")

# DeepSeek API Key（deepseek-v4-flash，用于生成排序理由）
# 获取地址：https://platform.deepseek.com/api_keys
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")

# ============================================================
# 以下为 A3 新增的可配置项（均支持环境变量覆盖，默认值匹配旧行为）
# ============================================================

# 混合打分权重（默认 0.6/0.3/0.1，保 verify.py / 旧测试通过）
VECTOR_WEIGHT: float = _env_float("EASYSEARCH_VECTOR_WEIGHT", "0.6")
BM25_WEIGHT: float = _env_float("EASYSEARCH_BM25_WEIGHT", "0.3")
POPULARITY_WEIGHT: float = _env_float("EASYSEARCH_POPULARITY_WEIGHT", "0.1")

# BM25 参数
BM25_K1: float = _env_float("EASYSEARCH_BM25_K1", "1.5")
BM25_B: float = _env_float("EASYSEARCH_BM25_B", "0.75")
# 多字段 BM25 字段权重（name 最强信号，aliases 次之，route 含路径关键词，intro 长文本权重最低）
BM25_FIELD_WEIGHTS: dict[str, float] = {
    "name": 3.0,
    "aliases": 2.0,
    "intro": 1.0,
    "route": 1.5,
}

# DIN 历史注意力参数
DIN_HISTORY_THRESHOLD: int = _env_int("EASYSEARCH_DIN_THRESHOLD", "10")
DIN_HISTORY_WINDOW: int = _env_int("EASYSEARCH_DIN_WINDOW", "50")
DIN_RELEVANCE_WEIGHT: float = _env_float("EASYSEARCH_DIN_RELEVANCE", "0.7")
DIN_RECENCY_WEIGHT: float = _env_float("EASYSEARCH_DIN_RECENCY", "0.3")
DIN_MIX_RATIO: float = _env_float("EASYSEARCH_DIN_MIX", "0.8")

# MMR 多样性重排（A5）：默认 0.85 轻微多样性，1.0 完全关闭
MMR_LAMBDA: float = _env_float("EASYSEARCH_MMR_LAMBDA", "0.85")

# popularity 时间衰减（A4）：tau=2592000（30天），window=90 天扫描窗口
POPULARITY_TAU: float = float(os.getenv("EASYSEARCH_POPULARITY_TAU", "2592000"))
POPULARITY_WINDOW_DAYS: int = _env_int("EASYSEARCH_POPULARITY_WINDOW_DAYS", "90")

# 同义词扩展（A1）与拼写纠错（A6）开关
SYNONYM_ENABLED: bool = _env_bool("EASYSEARCH_SYNONYM", "1")
SPELL_ENABLED: bool = _env_bool("EASYSEARCH_SPELL", "1")
SPELL_MAX_DISTANCE: int = _env_int("EASYSEARCH_SPELL_MAX_DIST", "2")

# ============================================================
# M2 异步化 / reason 可选化
# ============================================================
# DeepSeek reason 默认开启：有 API Key 时由 LLM 生成差异化排序理由。
# effort 默认 low（~0.5–1s）；无 Key / 失败自动降级到模板理由，不阻塞主链路。
REASON_ENABLED: bool = _env_bool("EASYSEARCH_REASON_ENABLED", "1")
# reason 推理强度：low（默认，~0.5–1s）/ medium / high（2–8s，不推荐线上默认）
REASON_EFFORT: str = os.getenv("EASYSEARCH_REASON_EFFORT", "low")

# 异步 HTTP 客户端连接池大小（M2：rerank + reason 并发用 httpx.AsyncClient）
ASYNC_HTTP_POOL_SIZE: int = _env_int("EASYSEARCH_ASYNC_HTTP_POOL", "32")

# ============================================================
# 结果缓存（Redis 可选降级）
# ============================================================
# Redis 连接 URL（配则启用 Redis 缓存，TTL=5min；未配走进程内存缓存 60s）
REDIS_URL: str = os.getenv("REDIS_URL", "") or os.getenv("EASYSEARCH_REDIS_URL", "")
# 缓存 TTL（秒，Redis 模式用此值，默认 300=5min；Memory 模式固定 60s 保旧行为）
CACHE_TTL: float = _env_float("EASYSEARCH_CACHE_TTL", "300")

# ============================================================
# M13 相关性提升
# ============================================================
# 负反馈（dwell time）：点后快速跳出视为负样本，对服务降权
NEGATIVE_FEEDBACK_ENABLED: bool = _env_bool("EASYSEARCH_NEG_FEEDBACK", "1")
# 快速跳出阈值（毫秒）：dwell < 此值记为负样本（默认 3000ms = 3 秒）
QUICK_BOUNCE_MS: int = _env_int("EASYSEARCH_QUICK_BOUNCE_MS", "3000")
# 单次负样本对 popularity_raw 的惩罚系数（归一前从 popularity 中扣除）
NEGATIVE_PENALTY: float = _env_float("EASYSEARCH_NEG_PENALTY", "0.5")
# 归一化模式：minmax（默认，向后兼容）/ rank（rank 归一，抗极值）/ zscore
NORMALIZE_MODE: str = os.getenv("EASYSEARCH_NORMALIZE_MODE", "minmax")
# 是否对向量分归一化后再参与 0.6/0.3/0.1 加权（默认关闭以保旧行为）
NORMALIZE_VECTOR: bool = _env_bool("EASYSEARCH_NORMALIZE_VECTOR", "0")
