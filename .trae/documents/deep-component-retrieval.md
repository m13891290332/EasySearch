# 深度组件检索：top-10 结果右侧显示最佳组件

## Context

用户要求新增「深度检索」选项：对 top-10 搜索结果，逐个获取对应服务的 route 页面，分析页面组件中与 query 最相关的信息，把最契合 query 的单个组件直接显示到每个搜索结果**右侧**，可点击。

**关键现状（已探查）**：
- 默认 KB（[services_dict_50.json](file:///d:/EasySearch/services_dict_50.json)）route 是相对路径（`/go/bank/...`，无 host，**后端不可抓取**），且 **`components` 字段为空**；最丰富的服务信息是 `service_intro` 的「使用方法」步骤。
- M15 `_maybe_deep_search`（[engine.py:817](file:///d:/EasySearch/easysearch/engine.py#L817)）做的是 query 扩展 + 二次检索 + RRF 融合并打 `deep_searched` 标签，**不抓取页面、不分析组件** —— 本需求是全新能力。
- M8 `execute_component_action`（[engine.py:1849](file:///d:/EasySearch/easysearch/engine.py#L1849)）按 KB `components` 白名单校验后 echo；默认 KB components 为空时返回 None。

**用户已确认的设计选择**：
1. 数据源 = **真抓取 + 降级**：http(s) 真实路由后端抓取页面 HTML（SSRF 防护）解析；相对路由降级为分析 `service_intro` 使用步骤 + KB `components` + 路由动作。当前演示 KB 走降级路径。
2. 点击行为 = **执行或跳转**：最佳组件有 `component`+`action` → 调 `/api/action/execute`；否则新窗口打开该服务 route。

**环境约束**（项目记忆）：本环境 Shell 不可用，测试需用户在外部终端运行。向后兼容硬约束：`engine.search` 仍返回 `list[dict]`；现有 DOM id/类名不变，新增为辅。

## 实现方案

### 1. 后端：SSRF 安全页面抓取（新 `easysearch/page_fetcher.py`）
- `fetch_page_async(url) -> str`：仅 http/https；解析 host → 拒绝 IP 字面量/私网/环回/链路本地（10.x、172.16-31、192.168、127、169.254、::1、fc00::/7）与 `localhost`；`httpx.AsyncClient`（timeout 5s、max_redirects 1、上限 512KB、Accept text/html）；失败/超时/被拦 → 返回 `""`。
- HTML→文本：复用 [safety.strip_html](file:///d:/EasySearch/easysearch/safety.py#L97)（剥标签+转义），再压空白、截 4000 字符喂 LLM。
- 模块级 lazy client + `close_page_fetcher()`，在 [api/main.py](file:///d:/EasySearch/api/main.py) lifespan shutdown 调用（镜像 `aclose_async_client` 模式）。

### 2. 后端：组件分析器（新 `easysearch/deep_components.py`）
- `ComponentAnalyzer(client)`（复用 DeepSeek/DashScope client 的 `post_json_async`，享 M12 重试）：
  - `analyze_async(query, service, page_text) -> dict`：构造 prompt（query + service_name + page_text/intro + 该服务真实 `components` 名单），要求返回严格 JSON `{label, reason, component, action}`；`component/action` 仅在命中该服务 KB components 名单时填，否则空串；`label`=简短可点击 CTA；`reason`=为何最契合 query。JSON 解析失败/LLM 不可用 → 走启发式。
  - `pick_heuristic(query, service) -> dict`（离线/降级）：若 `components` 非空 → 选 name/action 与 query token 重叠最多的；否则 label=route_info 的 `decision_button`（默认「进入服务」），component/action 空，href=route。
- 复用 [models.route_info](file:///d:/EasySearch/easysearch/models.py#L125) 规范化 route（dict/string）取 path/component/action_button；复用 [safety.validate_route_url](file:///d:/EasySearch/easysearch/safety.py#L106) 判定是否 http(s) 可抓取。

### 3. 后端：engine 编排（`easysearch/engine.py`）
- 新增 `analyze_deep_components_async(user_id, query, service_ids) -> list[dict]`：
  - 限 top-10（超出截断）；逐 id 查 `self.services`；`route_info(route)` → 若 path 为 http(s) 则 `fetch_page_async` 得 `page_text`，否则 `page_text=service_intro`（已含使用步骤）。
  - `asyncio.gather` 并发对 ≤10 个服务调 `ComponentAnalyzer.analyze_async`（LLM 不可用自动降级 `pick_heuristic`）。
  - 返回 `[{service_id, label, reason, component, action, href, route, source}]`（source="page"|"kb"|"fallback"）；异常静默单条降级。
- 复用 `self.metrics.record_external` 记 LLM/抓取耗时；`__init__` 注入 `ComponentAnalyzer`。

### 4. 后端：API 端点（`api/schemas.py` + `api/main.py`）
- schemas：`DeepComponentRequest{user_id, query, service_ids:list[str]}`、`DeepComponentItem{service_id,label,reason,component="",action="",href="",route="",source=""}`、`DeepComponentResponse{items:list[DeepComponentItem]}`。
- `POST /api/search/deep-components`（镜像 session 端点模式）：空 KB 409；`PromptInjectionError` 400；调 `engine.analyze_deep_components_async`；返回 `DeepComponentResponse`。

### 5. 前端：深度检索开关 + 右侧组件（`index.html` + `app.js` + `styles.css`）
- [index.html](file:///d:/EasySearch/frontend/index.html)：search-box 加 `<label class="deep-toggle"><input id="deep-toggle" type="checkbox"> 深度检索</label>`。
- [app.js](file:///d:/EasySearch/frontend/app.js) `renderResults`：把卡片内容包进 `<div class="card-main">…</div>` 并追加空 `<aside class="card-deep" data-sid="…"></aside>`（空时 CSS 隐藏）。
- 新增 `doDeepComponents(results)`：toggle 开且有结果时，取 top-10 `service_id` → `postJson('/api/search/deep-components', {user_id, query, service_ids})` → 对每张卡填 `.card-deep`：可点击 chip（`label` + `title=reason` + source 徽章）。点击：`component&&action` → 复用 `executeAction(sid,component,action)` 就地展示结果；否则 `isSafeRoute(href)` → 新窗口打开。
- 多条件搜索结果同样支持（`doMultiConditionSearch` 调 `renderResults` 后，toggle 开则同样触发 `doDeepComponents`）。
- [styles.css](file:///d:/EasySearch/frontend/styles.css)：`.card{display:flex;gap:14px;align-items:flex-start}` `.card-main{flex:1;min-width:0}` `.card-deep{width:220px;flex-shrink:0;border-left:1px dashed var(--border-solid);padding-left:14px}` `.card-deep:empty{display:none}` + `.deep-chip`（玻璃小卡，hover 高亮）+ `.deep-loading` 占位 + 响应式（窄屏 `.card` 改纵向、`.card-deep` 全宽）。

### 6. 测试（新 `tests/test_deep_components.py`）
- PageFetcher SSRF 守卫：拒绝 `http://127.0.0.1`、`http://localhost`、`http://10.0.0.1`、`javascript:...`；接受公网 http（用 monkeypatch 拦截真实网络）。
- ComponentAnalyzer 启发式：相对 route + 空 components + 无 API key → item `href=route`、`label=action_button 或「进入服务」、component/action 空。
- engine `analyze_deep_components_async`：mock client 返回 JSON → items 对齐 service_ids；异常单条降级不抛。
- API `POST /api/search/deep-components`：200 + items；空 KB 409；注入 400。
- 复用 `_make_engine` 模式（透传 `db_path`，见项目记忆 .npz 污染教训）。**未实跑**（Shell 不可用），用户外部终端验证。

## 关键文件

| 文件 | 改动 |
|---|---|
| [easysearch/page_fetcher.py](file:///d:/EasySearch/easysearch/page_fetcher.py) | 新建：SSRF 安全异步抓取 + HTML→文本 |
| [easysearch/deep_components.py](file:///d:/EasySearch/easysearch/deep_components.py) | 新建：ComponentAnalyzer（LLM + 启发式降级） |
| [easysearch/engine.py](file:///d:/EasySearch/easysearch/engine.py) | +`analyze_deep_components_async` 编排 + 注入 analyzer |
| [api/schemas.py](file:///d:/EasySearch/api/schemas.py) | +DeepComponent 三个模型 |
| [api/main.py](file:///d:/EasySearch/api/main.py) | +`POST /api/search/deep-components` + lifespan 关闭 fetcher |
| [frontend/index.html](file:///d:/EasySearch/frontend/index.html) | +深度检索开关 |
| [frontend/app.js](file:///d:/EasySearch/frontend/app.js) | renderResults 拆 card-main/card-deep + `doDeepComponents` + 点击执行/跳转 |
| [frontend/styles.css](file:///d:/EasySearch/frontend/styles.css) | 卡片 flex 双栏 + deep-chip 样式 + 响应式 |
| [tests/test_deep_components.py](file:///d:/EasySearch/tests/test_deep_components.py) | 新建：SSRF + 启发式 + 端点闭环 |

## 验证（用户在外部终端执行）
1. `python -m pytest tests/test_deep_components.py -v`
2. `python -m pytest tests/ -v`（确认无回归）
3. `uvicorn api.main:app --reload` → http://127.0.0.1:8000/
   - 勾选「深度检索」→ 搜索后每张结果卡右侧出现最佳组件 chip（演示 KB 走降级：label≈「进入服务」/动作按钮，点击新窗口打开 route）。
   - 配真实 DASHSCOPE/DEEPSEEK API Key + 含 http(s) 路由的 KB → 右侧 chip 为 LLM 据页面+query 挑出的组件，点击有 action 的执行 /api/action/execute。
   - DevTools Network：`POST /api/search/deep-components` 200，items 与 top-10 对齐。
4. `python verify.py`（兼容契约不破）
