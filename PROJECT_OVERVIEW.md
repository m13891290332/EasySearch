# EasySearch 项目说明文档

> 平台内服务搜索引擎：通过服务知识库检索「可点击服务访问路径 / 页面组件 / 决策执行按钮」。
> FastAPI 后端 + 前端分离搜索主页，全量向量 + BM25 混合检索，qwen3-vl-rerank 重排，全链路监控告警与降级。

---

## 一、项目介绍

### 1.1 项目定位

EasySearch 是一个面向**平台内服务**的智能搜索引擎。用户输入自然语言查询后，系统从服务知识库中检索最匹配的服务，并直接返回：

- **可点击的访问路径**（route）
- **页面组件**（component）
- **决策执行按钮**（decision_button）
- **排序理由**（rerank_reason）

让用户不仅能「搜到服务」，更能「直接进入服务的对应操作入口」。

### 1.2 核心价值

| 维度 | 说明 |
|------|------|
| **混合检索** | 向量语义相似度 + BM25 关键词匹配 + 热门性加权（0.6 / 0.3 / 0.1）|
| **意图驱动** | 6 种意图分类（导航型 / 多条件 / 指引型 / 信息型 / 会话型 / 默认），不同意图走不同检索流水线 |
| **大模型增强** | qwen3.7-text-embedding 向量化、qwen3-vl-rerank 重排、deepseek-v4-flash 生成排序理由与步骤化答案 |
| **离线降级** | 无 API Key 时全链路自动降级（本地向量 + 关键词重排 + 模板理由），仍可演示 |
| **生产级运维** | Prometheus 指标、SSE 实时大盘、告警规则、结构化日志、API Key 鉴权、限流、SSRF 防护 |

### 1.3 技术栈

- **后端**：Python 3.10+ / FastAPI / Uvicorn / Pydantic
- **存储**：SQLite（标准库 sqlite3，无需额外服务）
- **检索**：jieba 中文分词 / numpy BM25 倒排索引 / FAISS 向量索引（IndexFlatIP）
- **外部模型**：DashScope（qwen3.7-text-embedding / qwen3-vl-rerank）、DeepSeek（deepseek-v4-flash）
- **缓存**：进程内 LRU（默认）/ Redis（可选，记录最近 5 分钟 query + 结果 + 理由复用）
- **监控**：prometheus_client（可选软依赖）/ structlog（可选软依赖）
- **前端**：零构建原生 HTML/CSS/JS（液态玻璃风格），EventSource 订阅 SSE 实时大盘
- **HTTP**：urllib（同步）/ httpx（异步，连接池复用）

> 所有非标依赖（jieba / numpy / faiss / redis / prometheus_client / structlog / python-dotenv）均为**软依赖**：安装则用，未安装自动降级，核心功能不受影响。

---

## 二、功能介绍

### 2.1 检索能力

#### 检索模式（retrieval_mode）

| 模式 | 路径 | 适用场景 |
|------|------|----------|
| `hybrid`（默认）| 向量 + BM25 + 热门性 → rerank + 理由 | 通用语义 + 关键词混合检索 |
| `keyword` | 仅 BM25 + 热门性，跳过 rerank | 精确关键词匹配，毫秒级响应 |
| `semantic` | 仅向量 + 热门性，跳过 rerank | 语义相似度检索 |

#### 意图识别与路由（M5）

系统对 query 做规则意图分类，**优先级从高到低**：

| 意图 | 触发条件 | 路由行为 |
|------|----------|----------|
| `navigational` | 精确命中 service_name / alias | 跳过 MMR，直达唯一服务（即使未进 Top-10 也前置）|
| `multi_condition` | 含连接词（和/且/并且/同时/+ 与/及）| 走 M6 多条件交集检索 |
| `guide` | 含指引型短语（如何开始/新手/流程/步骤/怎么操作…）| 走 M16 步骤化答案模式 |
| `informational` | 含疑问词（怎么/如何/是什么/为什么…）| 正常检索 + M15 置信度评估 |
| `conversational` | 会话上下文存在 | 走 M7 长程对话 |
| `default` | 其余 | 标准 hybrid 检索 |

#### 高级搜索

- **多条件交集（M6）**：按连接词切分子查询，各子查询独立召回 Top-30 求交集。交集非空按首子查询保序；空交集降级 RRF 融合各子查询 Top-30。
- **长程对话（M7）**：基于 session_id 的多轮搜索。首轮宽召回 Top-40；后续轮用 session-level DIN 融合历史 query embedding，对累积候选集精化重排。支持 rollback 撤回上一轮。
- **泛化需求组合回复（需求2）**：对步骤型 query 并发检索各步 top1，组装卡片包组。
- **深度组件检索（需求3）**：对 top-10 结果，并发抓取每个服务 route 页面（SSRF 安全），LLM 分析「最契合 query 的页面组件」，在结果右侧渲染可点击组件 chip。

#### 相关性增强（A 组 / M13）

- **同义词扩展（A1）**：领域词典 + KB 动态抽取 alias ↔ service_name，BM25 路径追加同义词 token，向量路径归一到规范词
- **多字段 BM25（A2）**：name(3.0) / aliases(2.0) / route(1.5) / intro(1.0) 加权
- **时间衰减热门性（A4）**：tau=30 天衰减，window=90 天扫描，替代 raw count
- **MMR 多样性重排（A5）**：λ=0.85 在 rerank 后从 Top-20 选 Top-10
- **拼写纠错（A6）**：Levenshtein OOV 纠错（max_distance=2）
- **负反馈（M13）**：点后快速跳出（dwell < 3s）视为负样本，对服务降权

#### 二次深度检索（M15）

首次检索置信度不足时，自动触发二次深度检索（仅一次，防递归）：

触发条件（满足任一，navigational/multi/conversational 不触发）：
- 头部分离不足：top1 - top2 综合分差 < 0.05
- 命中稀疏：score>0 的结果数 < 3
- 信息型低相关：informational 且 top1 < 0.4

触发后：query 扩展（base + 同义词 + KB 共现词）→ 重检索 Top-30 → RRF 融合，结果项打 `deep_searched` 标签。

### 2.2 知识库管理（M9）

| 功能 | 端点 | 说明 |
|------|------|------|
| 导入版本 | `POST /api/kb/import` | 重建索引 → 落快照文件 → 写 kb_versions 元数据 → 置 active → 失效缓存 |
| 导出当前 | `GET /api/kb/export` | 导出为 JSON 条目列表（与输入同构，re-import hash 一致）|
| 版本列表 | `GET /api/kb/versions` | 列出全部版本快照（新→旧）|
| 回滚版本 | `POST /api/kb/rollback` | 读快照 → 重建索引 → 置 active |
| Embedding 状态 | `GET /api/kb/embedding-status` | total / embedded / in_progress / kb_hash / last_error |

内容寻址持久化：KB 内容 hash（基于 service_id + searchable_text 的 sha256）作为向量 npz 持久化 key，重启不重 embed。

### 2.3 监控告警（M10 / M14）

- **实时大盘**：`/dashboard` 页面，SSE 订阅 `/api/metrics/stream`，1s 刷新。展示 QPS / 错误率 / 缓存命中率 / 降级计数 / P95 / DB 池占用 + 各阶段 P50/P95/P99 + 外部调用健康度。
- **健康检查**：`/api/health` 返回服务数、模型配置、最近 100 次成功率/P95/缓存命中率 + 外部调用健康度。
- **Prometheus 抓取**：`/metrics` 端点（prometheus_client 可用走标准 generate_latest，否则手写 exposition 文本）。
- **理由流式**：`/api/reason` SSE 端点，排序理由按字符块增量推送。

### 2.4 前端能力

- **搜索主页**（`/`）：搜索框 + 自动补全 + 检索模式选择 + 高级搜索（+/- 行）+ 会话模式 + 深度检索开关
- **知识库管理页**（`/kb`）：导入 / 进度 / 版本列表 / 回滚 / 导出
- **实时大盘页**（`/dashboard`）：SSE 实时性能监控
- **设置面板**（右上角齿轮）：账号 / 默认检索模式 / 主题（亮色随机渐变 / 暗色动态着色器 / 自定义背景图），尊重系统「减弱动态效果」
- **液态玻璃风格**：低透明填充 + 斜角高光 + 边缘折射，光标悬停描边跟随点亮，标题扫光动画

---

## 三、项目架构

### 3.1 目录结构

```
EasySearch/
├── easysearch/               核心库
│   ├── config.py             配置中心（env 覆盖，源码零密钥）
│   ├── safety.py             M1 安全基座（输入清洗/注入防御/路由校验/LLM 输出校验）
│   ├── models.py             ServiceRecord + route 派生（component/decision_button）
│   ├── embedding.py          qwen3.7-text-embedding（批量 + 离线 fallback）
│   ├── vector_index.py       FAISS IndexFlatIP 向量索引
│   ├── bm25.py               多字段 BM25 倒排索引（jieba + numpy 批算）
│   ├── din.py                DIN 历史序列注意力（阈值 > 10 触发）
│   ├── reranker.py            qwen3-vl-rerank 重排 + deepseek 排序理由 + 单调性校验
│   ├── intent.py             M5 意图识别 + M15 置信度评估
│   ├── query_classifier.py   需求1 DeepSeek 语义预分类
│   ├── guide.py               M16 步骤化答案生成
│   ├── suggest.py            搜索框补全建议
│   ├── spell.py              A6 拼写纠错
│   ├── synonyms.py           A1 同义词扩展
│   ├── mmr.py                A5 MMR 多样性重排
│   ├── cache.py              结果缓存（Memory LRU + Redis，含 retrieval_mode key）
│   ├── store.py              SQLite 持久化（8 张表）
│   ├── metrics.py            M10/M14 MetricsCollector 单例 + Prometheus
│   ├── alerts.py             M10 AlertChecker 告警规则
│   ├── logging_config.py     M10 结构化日志（JsonFormatter + structlog 可选）
│   ├── page_fetcher.py       深度检索 SSRF 安全页面抓取
│   ├── deep_components.py    深度检索组件 LLM 分析
│   ├── dashscope.py          DashScope HTTP 客户端（重试 + 埋点）
│   ├── deepseek.py           DeepSeek 客户端（复用 dashscope）
│   ├── utils.py              tokenize / normalize_scores / extract_json
│   └── engine.py             ServiceSearchEngine 编排器（核心）
├── api/                      FastAPI 后端
│   ├── schemas.py            Pydantic 模型（30+ Request/Response）
│   ├── auth.py               M12 三中间件（ApiKey/BodySize/RateLimit）
│   ├── metrics.py            M14 实时大盘路由（realtime + SSE stream）
│   └── main.py               路由总装 + lifespan + 静态挂载
├── frontend/                 前端（零构建原生 JS）
│   ├── index.html / app.js / styles.css    搜索主页
│   ├── kb.html / kb.js                     知识库管理
│   ├── dashboard.html / dashboard.js      实时大盘
│   └── theme.js                            主题切换
├── tests/                    单元 + 集成测试
│   ├── conftest.py           测试隔离（env 清理 + 内存库）
│   └── test_*.py             ~15 个测试文件
├── docs/IMPLEMENTATION_PLAN.md   实施计划
├── plan.md                   M1-M16 模块定义
├── services_dict_50.json     默认知识库
├── verify.py                 端到端验证脚本
├── bench_bm25.py             BM25 基准测试
├── requirements.txt          依赖
└── .env.example              环境变量模板
```

### 3.2 分层架构

```
┌─────────────────────────────────────────────────────────┐
│  前端层（frontend/）                                       │
│  搜索主页 / 知识库管理 / 实时大盘 / 主题                    │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP + SSE
┌──────────────────────────▼──────────────────────────────┐
│  API 层（api/）                                           │
│  FastAPI 路由 + Pydantic 模型 + 三中间件 + lifespan        │
│  ┌──────────┬──────────┬──────────┐                     │
│  │限流(429) │体积(413) │鉴权(401)│ ← M12 安全中间件栈    │
│  └──────────┴──────────┴──────────┘                     │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  引擎编排层（engine.py ServiceSearchEngine）              │
│  意图路由 → 召回 → rerank → MMR → 深度检索 → 缓存 → 日志  │
│  监控埋点 + 告警评估                                       │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  核心组件层（easysearch/）                                │
│  BM25 / VectorIndex / DIN / Reranker / Intent / Cache    │
│  Safety / Metrics / Alerts / Store / PageFetcher        │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  外部依赖层                                               │
│  DashScope(qwen3 embedding/rerank) / DeepSeek(reason)    │
│  SQLite / Redis(可选) / FAISS / jieba / numpy            │
└─────────────────────────────────────────────────────────┘
```

### 3.3 模块全景（M1-M16）

| 模块 | 名称 | 核心交付 |
|------|------|----------|
| M1 | 安全基座 | 源码零密钥、提示词注入防御、特殊字符清洗、route URL 校验、LLM 输出校验 |
| M2 | 异步化 | httpx.AsyncClient 连接池、rerank/reason 并发 gather、reason 开关（默认关，effort=low）|
| M3 | BM25 倒排化 | posting list + numpy 批算，大库 BM25 Top-100 ∪ 向量 Top-100 候选集裁剪 |
| M4 | 缓存与持久化 | 向量 npz 持久化、embedding LRU、DIN 历史缓存、结果缓存（点击失效）|
| M5 | 意图识别 | IntentRouter 六分类 + navigational 直达置顶 |
| M6 | 多条件交集 | 子查询独立召回求交集，空交集 RRF 降级 |
| M7 | 长程对话 | session_id 多轮、session-level DIN 融合、累积候选精化、rollback |
| M8 | 页面内组件 | KB schema 扩展 components、结果卡片渲染组件动作、`/api/action/execute` |
| M9 | 知识库管理 | import/export/versions/rollback/embedding-status，内容寻址快照 |
| M10 | 监控告警 | MetricsCollector 单例、Prometheus 镜像、AlertChecker 四规则、结构化日志 |
| M11 | 数据日志 | search_logs / kb_op_logs，user_id 哈希化，无点击/高延迟/降级聚合 |
| M12 | 错误处理 | 重试分类（5xx 重试/4xx 不重试）、单调性校验、三中间件、前端错误码化 |
| M13 | 相关性提升 | 归一化（minmax/rank/zscore）、同义词挖掘、拼写纠错、负反馈降权 |
| M14 | 实时大盘 | realtime_summary 60s 窗口、SSE 推送、单请求 timing 字段 |
| M15 | 二次深度检索 | 置信度评估、query 扩展、Top-30 + RRF 融合（仅一次）|
| M16 | 答案模式 | guide 意图 → LLM 步骤化答案、service_id 白名单、answer_guide 响应 |

---

## 四、数据流图

### 4.1 主搜索流程（hybrid 模式）

```
用户 query
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ API 层（GET /api/search?user_id&query&retrieval_mode&session_id）│
│  限流 → 体积 → 鉴权 → 路由                                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ engine.search_async                                          │
│  ① sanitize_query（清洗 + 注入检测，命中→400）              │
│  ② store.append_query（记录查询历史）                        │
│  ③ 结果缓存查询（key = user_id + sha256(query) + mode）      │
│     命中 → 直接返回（记 cache_hit）                          │
└──────────────────────────┬───────────────────────────────────┘
                           │ 未命中
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ _build_top_candidates（召回 Top-20）                         │
│  ┌─ 向量路径：synonym normalize → embed → DIN(历史>10时融合)  │
│  ├─ BM25 路径：tokenize → 同义词扩展 → 拼写纠错              │
│  │              → batch_score_tokens（多字段加权）            │
│  ├─ 热门性：popularity_decayed（30天衰减 + 负反馈降权）      │
│  └─ 混合打分：0.6·向量 + 0.3·BM25 + 0.1·热门性（归一化）     │
│  大库(>200)：候选集 = BM25 Top-100 ∪ 向量 Top-100            │
└──────────────────────────┬───────────────────────────────────┘
                           │ Top-20 候选
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ rerank + reason 并发（asyncio.gather）                       │
│  ┌─ qwen3-vl-rerank 重排（失败→本地 token-overlap 降级）     │
│  └─ deepseek-v4-flash 排序理由（默认关闭→模板理由）          │
│     单调性校验：前半不应含负面词，后半不应含强正面词          │
└──────────────────────────┬───────────────────────────────────┘
                           │ 重排后 Top-20 + 理由
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ MMR 多样性选择（λ=0.85）→ Top-10                            │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 意图路由                                                     │
│  classify_intent（规则六分类）                               │
│  ├─ navigational → _pin_navigational_to_top（直达置顶）       │
│  ├─ multi_condition → 走 M6 交集（独立流水线）               │
│  ├─ guide → 走 M16 答案模式（独立流水线）                    │
│  ├─ informational/default → _maybe_deep_search（M15）        │
│  └─ conversational → 走 M7 会话（独立流水线）                │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 结果缓存写入（TTL：Memory 60s / Redis 300s）                 │
│ metrics.record_search（阶段计时 + 缓存/降级/错误标记）        │
│ _evaluate_and_fire_alerts（告警评估）                        │
│ _append_search_log（落 search_logs，user_id 哈希化）         │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 前端结果页                                                   │
│  可点击路径 / 页面组件 / 决策按钮 / 排序理由                 │
│  （深度检索开启时：右侧渲染组件 chip）                       │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 行为反馈闭环

```
搜索结果展示 → 用户点击服务
    │
    ▼
POST /api/click {user_id, service_id}
    │
    ▼
store.append_click（记录点击）
    │  ├─ global_clicks 累计（→ 热门性 raw）
    │  ├─ 标记 deprecated（服务已下线时不污染 popularity）
    │  └─ user_clicks（→ 用户级点击）
    ▼
_result_cache_invalidate（失效该用户缓存）
_mark_search_log_click（回填 search_logs.clicked_sid）
    │
    ▼
下次搜索：popularity_decayed 影响混合打分
    │
    ▼
（M13）用户停留反馈 POST /api/feedback
    ├─ dwell > 3s → 正样本（无操作）
    └─ dwell < 3s → 负样本 → 服务 popularity 降权
```

### 4.3 监控数据流

```
每次搜索 / 外部调用
    │
    ├─ engine ──→ metrics.record_search（total_ms / stages / cache_hit / degraded / error / intent）
    ├─ dashscope/deepseek ──→ metrics.record_external（service / ok / latency_ms）
    │                        （重试循环结束后仅上报最终结果一次）
    ▼
MetricsCollector 单例（进程内）
    ├─ _events deque(100)     ← /api/health「最近100次」+ 告警评估
    ├─ _realtime_events deque(600) ← realtime_summary 60s 窗口
    └─ 聚合计数器 + _external（含 consecutive_fail）
    │
    ├─→ /api/metrics/realtime?window=N   JSON 聚合（QPS/错误率/缓存/降级/P50/P95/P99）
    ├─→ /api/metrics/stream              SSE 1s 推送
    ├─→ /metrics                         Prometheus exposition
    ├─→ /api/health                      health_summary
    └─→ AlertChecker.evaluate
            ├─ error_rate > 5%
            ├─ P95 > 1s
            ├─ cache_hit_rate < 30%
            └─ 外部连续失败 ≥ 5
                │
                ▼
            fire → ERROR 日志 + 可选 webhook
```

---

## 五、权限与安全

### 5.1 API 安全中间件（M12）

请求经过 **LIFO 中间件栈**，执行顺序 = 限流 → 体积 → 鉴权 → 路由：

| 中间件 | 规则 | 失败响应 | 开关 |
|--------|------|----------|------|
| `RateLimitMiddleware` | per-IP token bucket，默认 60 req/min | 429 + `Retry-After: 60` | `EASYSEARCH_RATE_LIMIT=0` 禁用 |
| `BodySizeLimitMiddleware` | POST/PUT/PATCH 按 Content-Length，默认 10MB | 413 | `EASYSEARCH_MAX_BODY_BYTES` |
| `ApiKeyMiddleware` | `/api/*` 要求 `X-API-Key` 头匹配 | 401（无 Key）/ 403（错 Key）| `EASYSEARCH_API_KEY` 未设则透传 |

**白名单豁免**（监控探针不被拦截）：
- 限流白名单：`/api/health`、`/metrics`、`/api/metrics/realtime`、`/api/metrics/stream`、`/api/search/autocomplete`
- 鉴权白名单：`/api/health`、`/metrics`、`/api/metrics/realtime`、`/api/metrics/stream`

> 设计原则：默认关闭鉴权（离线/测试场景照常运行）；所有拒绝路径返回明确状态码 + 通用化错误消息，**不回显后端细节**。

### 5.2 输入安全（M1 safety.py）

| 能力 | 实现 |
|------|------|
| 查询清洗 `sanitize_query` | 限长 200 + 剥控制/零宽字符 + 注入关键词检测 → 命中抛 `PromptInjectionError`（→400）|
| Prompt 清洗 `sanitize_for_prompt` | reranker/reasoner 拼 LLM prompt 前清洗，注入关键词替换为 `[filtered]`（不抛）|
| KB 字段清洗 `sanitize_text` | 剥控制/零宽字符，限长 2000 |
| 路由校验 `validate_route_url` / `safe_route` | 仅允许相对路径或 http(s)/mailto，拒绝 javascript:/data:/vbscript: 等 |
| HTML 剥离 `strip_html` | 剥 `<script>` 等标签 + HTML 实体转义 |
| LLM 输出校验 `validate_llm_output` | service_id 必须在 KB 白名单 + reason 剥 HTML + 限长 200 |

**提示词注入关键词**覆盖：忽略上述/无视之前/ignore previous/system:/`<|im_start|>`/现在你扮演/jailbreak 等中英文越狱话术。

### 5.3 SSRF 防护（page_fetcher.py）

深度组件检索抓取服务 route 页面时：
- 仅 http/https 协议（拒绝 javascript:/data:/file:）
- 拒绝 IP 字面量落在私网 / 环回 / 链路本地 / 未分配 / 保留段
- 拒绝 localhost 及 .local / .internal 主机名
- **不跟随重定向**（避免被 302 到内网）
- 5s 超时、仅 text/html、4000 字符截断
- 任何失败返回空串，主链路降级用 service_intro 做组件分析

### 5.4 隐私保护（M11）

- `search_logs.user_hash = sha256(user_id + salt)`，**不存原始 user_id**
- 盐值可由 `EASYSEARCH_USER_SALT` 覆盖（默认进程级）
- 其余行为表（user_queries/user_clicks）仍用明文 user_id（同 tenant 内可信）

### 5.5 密钥管理（M1）

- **源码零密钥**：所有 API Key 从环境变量 / `.env` 文件读取，不硬编码
- `.env` 已加入 `.gitignore`
- API Key 读取优先级：构造参数 > 环境变量 > 离线 fallback
- 无 Key 时全链路自动降级，仍可演示

---

## 六、告警体系

### 6.1 告警规则（AlertChecker）

基于 `MetricsCollector` 滚动窗口评估，**每次搜索后自动评估**（`_evaluate_and_fire_alerts`，失败静默不影响主链路）：

| 规则 | 阈值 | 级别 | 触发条件 |
|------|------|------|----------|
| `error_rate` | > 5% | ERROR | 最近窗口错误率超阈值（需 ≥5 次样本，防冷启动误报）|
| `p95_latency` | > 1000ms | WARN | P95 端到端延迟超阈值 |
| `cache_hit_rate` | < 30% | WARN | 缓存命中率低于阈值 |
| `external_consecutive_fail` | ≥ 5 | ERROR | DashScope/DeepSeek 连续失败次数（独立于样本量，立即告警）|

阈值均可经环境变量覆盖：
```
EASYSEARCH_ALERT_ERROR_RATE=0.05
EASYSEARCH_ALERT_P95_MS=1000
EASYSEARCH_ALERT_CACHE_HIT=0.30
EASYSEARCH_ALERT_EXT_FAIL=5
```

### 6.2 告警落地

- **ERROR/WARN 日志**：触发即落结构化日志（`ALERT rule=... value=... msg=...`）
- **可选 webhook**：设置 `EASYSEARCH_ALERT_WEBHOOK` 后，触发时 POST JSON `{"alerts": [...]}`（5s 超时，失败静默）
- **无告警静默**：不触发时不产生任何输出

### 6.3 监控指标（MetricsCollector）

进程内单例，双缓冲设计：

| 缓冲 | 容量 | 用途 |
|------|------|------|
| `_events` | deque(100) | `/api/health` 最近 100 次 + 告警评估 |
| `_realtime_events` | deque(600) | realtime_summary 60s 滚动窗口聚合（分离以保 health 语义不变）|

聚合计数器：`search_total` / `search_errors` / `cache_hits` / `cache_misses` / `_external`（含 consecutive_fail）/ `_kb_embedding_in_progress`。

**Prometheus 镜像**（prometheus_client 可用时）：
- `easysearch_search_total` / `easysearch_search_errors_total`（Counter）
- `easysearch_cache_hits_total` / `easysearch_cache_misses_total`（Counter）
- `easysearch_search_latency_seconds{stage}`（Histogram，分阶段）
- `easysearch_external_call_total{service,status}`（Counter）
- `easysearch_external_call_latency_seconds{service}`（Histogram）
- `easysearch_kb_embedding_in_progress`（Gauge）
- `easysearch_db_pool_usage`（Gauge，占位预留）

---

## 七、错误处理与降级（M12）

### 7.1 远程调用重试（dashscope/deepseek）

异常分类：
- `RetryableHTTPError`：5xx / 超时 / 网络异常 → **指数退避重试**（base=0.5s，默认 2 次）
- `NonRetryableHTTPError`：4xx 客户端错误 → 不重试，原样抛出便于上层降级
- 普通 RuntimeError → 不重试（保向后兼容）

埋点策略：**仅在重试循环结束后上报最终结果一次**，避免每次失败尝试计入 `_external` 导致 fail_rate 失真。

### 7.2 LLM 输出校验

- **JSON 解析重试**：排序理由首次解析为空 → 重试 1 次，仍失败降级模板（返回 `{}`）
- **rank 单调性校验**：top 前半 rank 的 reason 不应含负面词（次/较弱/无关…），后半不应含强正面词（最相关/最佳…），命中则删除该条回退模板

### 7.3 全链路降级矩阵

| 组件 | 正常 | 降级 |
|------|------|------|
| Embedding | qwen3.7-text-embedding | 离线 hash 向量（本地）|
| Rerank | qwen3-vl-rerank | 本地 token-overlap 重排（`score + 0.01*overlap`）|
| 排序理由 | deepseek-v4-flash（默认关闭）| 差异化模板理由（按命中字段：名称/别名/简介/语义）|
| 缓存 | Redis（5min）| 内存 LRU（60s）|
| Prometheus | prometheus_client | 手写 exposition 文本 |
| 日志 | structlog | stdlib JsonFormatter |
| 深度检索 | LLM 组件分析 | 启发式 pick_heuristic |
| guide 答案 | LLM 步骤化 | 降级 list 模式 Top-10 |

---

## 八、数据存储

### 8.1 SQLite 表结构（store.py）

| 表 | 用途 | 关键字段 |
|----|------|----------|
| `user_queries` | 用户查询历史（DIN 序列 + 下拉）| user_id, query, ts |
| `user_clicks` | 用户点击（含 deprecated 标记）| user_id, service_id, ts, deprecated |
| `global_clicks` | 服务总点击数（热门性 raw）| service_id, count |
| `search_sessions` | M7 会话轮次（支持 rollback）| session_id, turn_idx, query, top_ids_json |
| `service_feedback` | M13 负反馈（dwell time）| user_id, service_id, dwell_ms |
| `kb_versions` | M9 KB 版本快照元数据 | version_id, kb_hash, path, active |
| `search_logs` | M11 全链路搜索日志 | user_hash, query, intent, top_ids, latencies, cache_hit, degraded, clicked_sid |
| `kb_op_logs` | M11 KB 运维操作日志 | op, version_id, kb_hash, ok, detail_json |

并发安全：写操作加进程内锁，`check_same_thread=False`。uvicorn 默认单 worker 下足够；多 worker 部署需替换为连接池。

### 8.2 文件持久化

| 目录 | 内容 |
|------|------|
| `data/embeddings/` | 向量 npz（`emb_{kb_hash}.npz`，内容寻址，重启不重 embed）|
| `data/kb_versions/` | KB 版本快照 JSON（`{version_id}.json`）|
| `data/related/` | 相关服务 top-3 预计算（`related_{kb_hash}.json`，搜索框路由占位）|
| `data/easysearch.db` | SQLite 主库 |

---

## 九、API 总览

### 9.1 搜索与交互

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/search` | 主搜索（user_id/query/mode/retrieval_mode/session_id）→ SearchResponse |
| POST | `/api/search/intersection` | 多条件交集搜索 → Top-10 + match_mode |
| POST | `/api/search/deep-components` | Top-10 深度组件推荐 |
| POST | `/api/search/session` | 长程对话（search/rollback）|
| GET | `/api/search/suggest` | 搜索补全建议 |
| GET | `/api/search/autocomplete` | 自动补全（按键级，限流豁免）|
| POST | `/api/click` | 记录点击 |
| POST | `/api/feedback` | 停留反馈（dwell time）|
| GET | `/api/dropdown` | 首页下拉（最近3搜索/点击/最热3服务）|
| GET | `/api/service` | 服务详情 |
| POST | `/api/action/execute` | 组件动作执行（白名单校验）|
| GET | `/api/reason` | 排序理由 SSE 流式 |

### 9.2 知识库管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/knowledge-base/upload` | 上传 JSON 知识库 |
| POST | `/api/kb/import` | 导入版本（重建索引 + 快照）|
| GET | `/api/kb/export` | 导出当前 KB |
| GET | `/api/kb/versions` | 版本列表 |
| POST | `/api/kb/rollback` | 回滚版本 |
| GET | `/api/kb/embedding-status` | Embedding 状态 |

### 9.3 监控与日志

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查（服务数 + 监控摘要 + 外部健康度）|
| GET | `/metrics` | Prometheus exposition |
| GET | `/api/metrics/realtime?window=` | 实时聚合（1-600s）|
| GET | `/api/metrics/stream?interval=&max_events=` | SSE 实时推送 |
| GET | `/api/logs/search/no-click` | 无点击 query 聚合 |
| GET | `/api/logs/search/slow` | 高延迟 query 聚合 |
| GET | `/api/logs/degradation` | 降级/缓存频次统计 |
| GET | `/api/logs/search/recent` | 最近搜索日志 |
| GET | `/api/logs/kb-ops` | KB 操作日志 |

### 9.4 前端页面

| 路径 | 页面 |
|------|------|
| `/` | 搜索主页 |
| `/kb` | 知识库管理 |
| `/dashboard` | 实时性能大盘 |

---

## 十、配置与环境变量

### 10.1 关键配置（config.py，均支持环境变量覆盖）

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `VECTOR_WEIGHT` / `BM25_WEIGHT` / `POPULARITY_WEIGHT` | 0.6 / 0.3 / 0.1 | 混合打分权重 |
| `BM25_K1` / `BM25_B` | 1.5 / 0.75 | BM25 参数 |
| `BM25_FIELD_WEIGHTS` | name=3.0/aliases=2.0/route=1.5/intro=1.0 | 多字段 BM25 加权 |
| `DIN_HISTORY_THRESHOLD` | 10 | DIN 触发阈值（历史查询数）|
| `MMR_LAMBDA` | 0.85 | MMR 多样性参数（1.0 完全关闭）|
| `POPULARITY_TAU` | 2592000（30天）| 热门性时间衰减 |
| `REASON_ENABLED` | False | 排序理由开关（high effort 是 SLA 杀手）|
| `REASON_EFFORT` | low | 推理强度 |
| `CACHE_TTL` | 300（Redis）/ 60（Memory）| 缓存 TTL |
| `QUICK_BOUNCE_MS` | 3000 | 快速跳出阈值（负反馈）|

### 10.2 环境变量（.env.example）

| 变量 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | qwen3 embedding/rerank（必填以启用真实模型）|
| `DEEPSEEK_API_KEY` | deepseek-v4-flash 排序理由 |
| `EASYSEARCH_KB` | 知识库 JSON 路径（默认 services_dict_50.json）|
| `EASYSEARCH_API_KEY` | API Key 鉴权（留空关闭）|
| `EASYSEARCH_RATE_LIMIT` | per-IP 限流（60/min，0=禁用）|
| `EASYSEARCH_MAX_BODY_BYTES` | 上传体积上限（10MB）|
| `EASYSEARCH_ALERT_WEBHOOK` | 告警 webhook（留空仅日志）|
| `REDIS_URL` / `EASYSEARCH_REDIS_URL` | Redis 缓存（未配走内存）|
| `EASYSEARCH_CACHE_TTL` | Redis 缓存 TTL（300s）|

---

## 十一、运行与部署

### 11.1 安装

```bash
pip install -r requirements.txt
```

### 11.2 启动

```bash
# 离线模式（无 Key，全链路降级，仍可演示）
uvicorn api.main:app --reload

# 接入真实模型：复制 .env.example → .env 填入 Key
cp .env.example .env
uvicorn api.main:app --reload
```

浏览器打开 http://localhost:8000

### 11.3 测试

```bash
python -m pytest tests/ -v          # 全量测试
python verify.py                    # 端到端验证
python bench_bm25.py                # BM25 基准
```

---

## 十二、作为库使用

```python
from easysearch import DashScopeClient, ServiceSearchEngine

engine = ServiceSearchEngine(dashscope_client=DashScopeClient())
engine.upload_knowledge_base_from_json("services_dict_50.json")

# 混合检索
results = engine.search(user_id="u-1", query="开户", retrieval_mode="hybrid")
for item in results:
    print(item["service_name"], item["route"], item["rerank_reason"])

# 记录点击（影响下次排序的热门性）
engine.record_click("u-1", results[0]["service_id"])

# 首页下拉
print(engine.homepage_dropdown("u-1"))
```

---

## 十三、容量演进与扩展点

| 场景 | 当前 | 扩展方向 |
|------|------|----------|
| 多 worker 部署 | 进程内缓存 + 单 worker 锁 | Redis 共享缓存 + SQLite 连接池 |
| 大库（>10K）| BM25 Top-100 ∪ 向量 Top-100 候选裁剪 | FAISS IVF/HNSW 近似索引 |
| 限流 | 进程内 token bucket | Redis 分布式限流 |
| DB 池监控 | 占位 0 | 接入连接池 gauge |
| 外部告警 | ERROR 日志 + webhook | 接入 Alertmanager / 钉钉 / 飞书 |

---

*文档基于 M1-M16 全模块实现编写，覆盖项目介绍、功能、架构、数据流、权限、告警等核心内容，便于项目汇报与交接。*
