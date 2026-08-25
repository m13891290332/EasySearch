# EasySearch 实施计划文档

> 版本：v2.0（全量重构）  日期：2026-08-24
> 目标来源：`/goal 完善python语言的专门搜索某个平台内服务的搜索引擎…`（goal_id: 6a8c514be2d547f94aded45d，状态 active）

---

## 一、项目概述

### 1.1 项目目标
将现有 Python 服务搜索引擎从「单库 + 内存状态」升级为**独立可运行的搜索引擎产品**：FastAPI 后端 + 前端分离搜索主页，全量重写检索与重排链路，SQLite 落盘用户行为，离线 fallback 保证无 API Key 也可演示。

### 1.2 预期交付物
- `easysearch/` 核心库（重写：embedding / bm25+jieba / din / reranker / store / engine）
- `api/` FastAPI 后端（知识库上传、搜索、点击、下拉、健康检查）
- `frontend/` 独立搜索主页（下拉默认项 + 结果页含可点击路径/页面组件/决策执行按钮/排序理由）
- `data/easysearch.db` SQLite 持久化
- `tests/` 单元测试（重写）+ `requirements.txt` + 更新后的 `README.md`

### 1.3 关键约束（来自用户确认）
| 维度 | 选择 |
|------|------|
| 主页形态 | FastAPI + 前端分离 |
| 改进范围 | 全量重构检索与重排 |
| 持久化 | SQLite |
| API Key | 离线 fallback 优先（无 Key 本地降级，有 Key 调真实 Qwen） |

### 1.4 不变的核心规则（不可在重构中破坏）
- 知识库 5 字段：`service_id / service_name / aliases / service_intro / route`
- DIN 触发阈值：用户历史查询数 **> 10**
- 混合打分公式：`score = 0.6·向量相似度 + 0.3·BM25 + 0.1·热门性`
- 召回 → Top-20 → rerank → **Top-10**
- 模型：`qwen3.7-text-embedding` / `qwen3-vl-rerank` / `qwen3-vl-plus`
- 首页下拉：最近 3 次搜索词、最近 3 次点击服务、全局最热 3 个服务

---

## 二、需求拆解与优先级划分

### P0 — 计划与可行性（本文档）
- 交付：本计划文档 + 风险与冲突标注。标准：7 大模块齐全、任务可分配。

### P1 — 基础设施（前置依赖）
| 任务 | 交付标准 | 依赖 |
|------|----------|------|
| T1.1 `requirements.txt` | 含 fastapi/uvicorn/jieba/pydantic/pytest，版本钉版 | - |
| T1.2 目录结构 | `easysearch/ api/ frontend/ data/ tests/ docs/` 就位 | T1.1 |
| T1.3 `store.py` SQLite 层 | 表 `user_queries/user_clicks/global_clicks`；CRUD + 事务；并发安全（check_same_thread=False） | T1.2 |
| T1.4 DashScope 客户端迁移 | `dashscope.py`，保留 requester 注入便于测试 | - |

### P2 — 检索重排核心重写（最高价值）
| 任务 | 交付标准 | 依赖 |
|------|----------|------|
| T2.1 `models.py` | `ServiceRecord`；route 同时支持 dict 与 string，并派生 component/decision_button | - |
| T2.2 `embedding.py` | qwen3.7-text-embedding；**批量** embed；L2 归一；本地 hash fallback（dim 可配，默认 1024） | T1.4 |
| T2.3 `bm25.py` | **jieba 中文分词** + 正则英文；IDF/TF；min-max 归一接口 | T1.1 |
| T2.4 `din.py` | 历史序列注意力；阈值 >10 触发；recency+relevance 加权 | T2.2 |
| T2.5 `reranker.py` | qwen3-vl-rerank Top-20 重排 + qwen3-vl-plus 生成排序理由；本地 fallback（关键词重合） | T1.4 |
| T2.6 `engine.py` | 编排：上传→批量向量化→混合打分→Top-20→rerank→Top-10；记录查询/点击；下拉；route 派生 | T2.1-T2.5, T1.3 |

### P3 — FastAPI 后端
| 任务 | 交付标准 | 依赖 |
|------|----------|------|
| T3.1 `schemas.py` | Pydantic：SearchRequest/Response/ClickRequest/DropdownResponse | T2.6 |
| T3.2 `main.py` | 应用工厂；CORS；启动时加载 services_dict_50.json | T3.1 |
| T3.3 路由 | POST `/api/knowledge-base/upload`、GET `/api/search`、POST `/api/click`、GET `/api/dropdown`、GET `/api/health` | T3.2 |
| T3.4 错误处理 | 统一异常 → JSON；输入校验 | T3.3 |

### P4 — 前端分离搜索主页
| 任务 | 交付标准 | 依赖 |
|------|----------|------|
| T4.1 `index.html` | 独立搜索主页：搜索框 + 三段下拉 + 结果区 | T3.3 |
| T4.2 `app.js` | fetch 调 API；下拉默认项展示；结果渲染可点击 route/组件/决策按钮/排序理由 | T4.1 |
| T4.3 `styles.css` | 清晰可读样式，无外部 CDN 依赖（纯本地） | T4.1 |

### P5 — 测试与文档
| 任务 | 交付标准 | 依赖 |
|------|----------|------|
| T5.1 单元测试重写 | 覆盖 route 派生/混合公式/DIN 阈值/批量 embed/jieba 分词/SQLite 持久化/API payload | P2,P3 |
| T5.2 README | 运行命令（uvicorn / 前端 / 测试）、API Key 配置、离线模式说明 | P3,P4 |
| T5.3 可行性复核 | 对照本计划逐项核对，更新遗留项 | 全部 |

---

## 三、技术选型建议

| 层 | 选型 | 理由 |
|----|------|------|
| 后端框架 | FastAPI + Uvicorn | 原生 async、自动 OpenAPI、Pydantic 校验，贴合「前端分离」 |
| 前端 | 原生 HTML/CSS/JS（无构建） | 搜索主页交互轻量，零构建可跑；如需 SPA 后续可升级 Vite+React |
| 中文分词 | jieba | BM25 召回质量关键依赖，纯 Python 易安装 |
| 持久化 | sqlite3（标准库） | 单文件、零运维、多用户历史落盘 |
| 向量检索 | 内存矩阵 + cosine（Top-K） | 50 条规模无需 FAISS；保留扩展接口 |
| 模型 | qwen3.7-text-embedding / qwen3-vl-rerank / qwen3-vl-plus | 用户指定 |
| 离线降级 | hash 向量 + 关键词重合 rerank | 保证无 Key/断网可演示 |
| 测试 | pytest + unittest 兼容 | 既有用例迁移 |

---

## 四、分阶段实施里程碑

- **M1（P1+P2 核心）**：核心库重写完成，`pytest` 全绿，混合打分/DIN/rerank 链路单测通过。
- **M2（P3 后端）**：FastAPI 起服务，`curl` 五接口可用，启动自动装载 50 条知识库。
- **M3（P4 前端）**：浏览器打开主页 → 下拉默认项正确 → 搜索 → 点击 → 下拉更新。
- **M4（P5 收尾）**：测试覆盖 ≥ 既有用例数；README 可照跑；计划可行性复核完成。

---

## 五、风险预判与应对方案

| 风险 | 等级 | 应对 |
|------|------|------|
| 真实 Qwen API 需网络+Key，CI/本地常缺 | 高 | 离线 fallback 全链路可跑；测试默认走 requester 注入 mock |
| `services_dict_50.json` 的 route 是字符串，component/decision_button 缺失 | 高 | engine 派生：string route→可点击 path + 由 slug 派生 component + 默认决策按钮「进入」；dict route 仍走原提取 |
| SQLite 多线程（uvicorn 多 worker）并发写 | 中 | 默认单 worker；`check_same_thread=False` + 写操作加锁/事务 |
| 全量重构破坏既有 API 兼容 | 中 | 重写同步更新测试；保留 `ServiceSearchEngine.search/record_click/homepage_dropdown` 公共方法签名 |
| jieba 首次加载慢 | 低 | 模块级懒加载，首次调用预热 |
| 前端跨域 | 低 | FastAPI 开 CORS（localhost） |

---

## 六、质量保障措施

- **单元测试**：route 派生（string/dict 两路）、混合公式断言 `0.6a+0.3b+0.1c`、DIN 阈值 `>10`、批量 embed 形状、jieba 分词命中、SQLite 重启后历史不丢、API payload 模型名/端点正确。
- **mock 注入**：`DashScopeClient(requester=...)` 保留，测试不触网。
- **回归基线**：保留 `test_search_engine.py` 等价断言，重构后必须仍通过。
- **端到端冒烟**：M3 手动/脚本验证主页→搜索→点击→下拉更新闭环。
- **代码规范**：类型注解 + docstring；中文注释跟随用户语言。

---

## 七、资源需求估算

| 资源 | 估算 |
|------|------|
| 代码量 | 核心 ~600 行 / API ~200 行 / 前端 ~300 行 / 测试 ~250 行 |
| 依赖 | fastapi、uvicorn、jieba、pydantic、pytest（均为纯 Python 或易装） |
| 外部服务 | DashScope API（可选；无则降级） |
| 工时（机器执行） | M1-M4 顺序推进，单次会话内完成核心 + 后端 + 前端 + 测试 |

---

## 八、可行性校验与冲突标注

✅ **通过项**：5 字段模型、混合公式、Top-20→Top-10、DIN>10、三模型名、下拉三项——均可在现有架构 1:1 对应实现，无逻辑冲突。

⚠️ **需注意/潜在冲突**：
1. **「可点击路径/页面组件/决策执行按钮」对 string route 数据缺位**：真实 50 条数据 route 全为字符串，component/decision_button 无原始值。**决策**：由 engine 派生（slug→component、默认决策按钮），并在结果字段标注 `derived=true`，避免伪造数据误导。此为对需求的「补齐」而非破坏。
2. **「全量重构」与「保持公共 API 兼容」张力**：以**行为兼容**优先——`ServiceSearchEngine` 三大公共方法签名不变，内部实现全换；测试同步重写。
3. **DIN「历史超过 10 个」**：严格 `len(history) > 10` 触发，与原实现一致；当前 query 不计入历史序列参与注意力。
4. **前端「分离」与「无构建」取舍**：选原生 JS 零构建以保可跑性；若后续需 SPA 可平滑替换。
5. **SQLite 并发**：默认 uvicorn 单 worker；多 worker 部署需改用连接池或 PostgreSQL（超本计划范围，标注）。
