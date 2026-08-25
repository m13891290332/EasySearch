# EasySearch

> 平台内服务搜索引擎：通过服务知识库检索「可点击服务访问路径 / 页面组件 / 决策执行按钮」。
> FastAPI 后端 + 前端分离搜索主页，全量向量+BM25 混合检索，qwen3-vl-rerank 重排。

## 架构

```
知识库(services_dict_50.json) ── 倒排索引(BM25/jieba) + qwen3.7-text-embedding 批量向量化
                                          │
用户 query ── qwen3.7-text-embedding 向量化
        │── (历史查询 > 10 时) DIN 历史序列注意力优化 query 向量
        ▼
混合检索: score = 0.6·向量相似度 + 0.3·BM25 + 0.1·热门性  ──► Top-20
        ▼
qwen3-vl-rerank 重排 + qwen3-vl-plus 生成排序理由  ──► Top-10
        ▼
前端结果页：可点击路径 / 页面组件 / 决策执行按钮 / 排序理由
```

## 目录结构

```
easysearch/        核心库
  ├ dashscope.py    DashScope HTTP 客户端
  ├ models.py       ServiceRecord + route 派生（component/decision_button）
  ├ embedding.py    qwen3.7-text-embedding（批量 + 离线 fallback）
  ├ bm25.py         BM25 倒排索引（jieba 中文分词）
  ├ din.py          DIN 历史序列注意力（阈值 > 10）
  ├ reranker.py     qwen3-vl-rerank + qwen3-vl-plus 排序理由
  ├ store.py        SQLite 持久化（查询/点击/全局热门）
  └ engine.py       ServiceSearchEngine 编排
api/               FastAPI 后端
  ├ schemas.py      Pydantic 模型
  └ main.py         路由：搜索/点击/下拉/上传/健康检查 + 静态主页
frontend/          前端分离搜索主页（零构建原生 JS）
tests/             单元测试（核心 + API）
docs/IMPLEMENTATION_PLAN.md  实施计划文档
```

## 安装

```bash
pip install -r requirements.txt
```

> `jieba / fastapi / uvicorn / pydantic / pytest / httpx` 均为纯 Python 或易装。
> `sqlite3` 为 Python 标准库。无需安装 DashScope SDK。

## 运行

### 离线模式（无需 API Key，本地降级向量 + 关键词 rerank）

```bash
uvicorn api.main:app --reload
# 浏览器打开 http://localhost:8000
```

### 接入真实 Qwen 模型

**方式一：在配置文件中写死（推荐，无需环境变量）**

编辑 [easysearch/config.py](easysearch/config.py)，在引号内填入你的 Key：

```python
DASHSCOPE_API_KEY: str = "sk-你的实际Key"
```

保存后直接启动即可生效，无需任何环境变量设置。

**方式二：环境变量（兼容旧用法）**

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="你的Key"
uvicorn api.main:app --reload
```

**API Key 读取优先级**：构造参数 > `config.py` 写死值 > 环境变量 > 离线 fallback。

无 Key 时全链路自动降级，仍可演示；有 Key 时调用真实
`qwen3.7-text-embedding` / `qwen3-vl-rerank` / `qwen3-vl-plus`。

> ⚠️ 安全：填入 Key 后的 `config.py` 请勿提交到公开仓库，已加入 `.gitignore`。

**自定义知识库路径**：

```bash
$env:EASYSEARCH_KB="path/to/your_services.json"
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/api/health` | 健康检查：服务数、DashScope 是否配置 |
| GET  | `/api/search?user_id=&query=` | 搜索 → Top-10（含 route/component/decision_button/rerank_reason） |
| POST | `/api/click` | `{user_id, service_id}` 记录点击 |
| GET  | `/api/dropdown?user_id=` | 首页下拉：最近3搜索词 / 最近3点击服务 / 全局最热3服务 |
| POST | `/api/knowledge-base/upload` | 上传 JSON 数组知识库 |
| GET  | `/` | 搜索主页（前端） |

## 作为库使用

```python
from easysearch import DashScopeClient, ServiceSearchEngine

engine = ServiceSearchEngine(dashscope_client=DashScopeClient())
engine.upload_knowledge_base_from_json("services_dict_50.json")

results = engine.search(user_id="u-1", query="开户")
for item in results:
    print(item["service_name"], item["route"], item["component"], item["decision_button"])

engine.record_click("u-1", results[0]["service_id"])
print(engine.homepage_dropdown("u-1"))
```

## 知识库字段

`service_id` / `service_name` / `aliases` / `service_intro` / `route`

- `route` 为 dict：提取 `path` / `component` / `action_button`（原始）
- `route` 为 string：可点击 path；`component` 由 slug 派生（如 `/go/account/open-account` → `OpenAccount`）；决策按钮默认「进入」（结果中 `derived=true` 标注）

## 测试

```bash
pytest -q
# 或
python -m unittest discover -s tests -p 'test_*.py'
```

## 核心规则（与需求一致）

- 混合打分权重：`0.6 / 0.3 / 0.1`
- DIN 触发：用户历史查询数 `> 10`
- 召回 Top-20 → rerank → Top-10
- 模型：`qwen3.7-text-embedding` / `qwen3-vl-rerank` / `qwen3-vl-plus`
