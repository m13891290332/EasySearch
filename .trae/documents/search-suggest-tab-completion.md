# 搜索框灰色建议 + Tab 补全

## Context

EasySearch 首页搜索框目前只有静态 placeholder 和聚焦时的下拉（最近搜索/点击/热门）。用户希望获得 Chrome omnibox 风格的体验：**输入过程中在框内实时显示灰色补全建议**，建议由 DeepSeek 基于该用户的搜索历史 + 点击历史生成；用户按 **Tab** 键即可把建议自动填入 query 框。

这是纯新增功能，不触碰 `engine.search` 的 `list[dict]` 返回契约（verify.py 依赖），不影响现有 `/api/search`、`/api/dropdown` 等端点。

## 设计要点

- **后端**：仿 [guide.py](file:///d:/EasySearch/easysearch/guide.py) 的 `GuideGenerator` 模式新增 `QuerySuggester`，复用 `DeepSeekClient.post_json(_async)`。LLM 输出 JSON `{"completion":"..."}`，服务端硬性校验 `completion.startswith(partial)` 且更长，否则返回 None（宁可丢弃也不展示错误建议）。`thinking` 设为 `disabled`（per-keystroke 延迟敏感，区别于 guide 的 enabled）。
- **历史聚合**：复用 `store.recent_queries(user_id, 5)` + `store.recent_clicks(user_id, 5)`，service_id → service_name 用 `self.services[sid].service_name` + `if sid in self.services` 守卫（与 `homepage_dropdown` 一致，过滤已下线服务）。
- **前端**：`wrapper div + 绝对定位 overlay span` 方案实现灰色叠加。单个 `#query` 保留所有现有事件绑定；overlay span 设 `pointer-events:none` + `aria-hidden=true` 仅作视觉装饰。input 背景透明，其黑色文字自然遮盖 overlay 前 N 个字符（即 partial 部分），只露出灰色后缀。
- **防抖 200ms + 竞态防护**：`suggestRequestId` 单调递增丢弃过期响应；`input` 事件立即清旧建议再 schedule，防闪烁。
- **IME 兼容**：`compositionstart/end` 期间不发请求、清建议。
- **Tab 行为**：仅在有有效建议时 `preventDefault` 接受建议；无建议时 Tab 走默认（移焦点），保键盘可访问性。
- **降级**：所有异常路径（LLM 不可用/超时/前缀不匹配/解析失败）→ 返回空 completion，前端隐藏灰色文字。绝不让后端异常打断输入体验。

## 文件改动清单

| 文件 | 类型 | 摘要 |
|------|------|------|
| [easysearch/suggest.py](file:///d:/EasySearch/easysearch/suggest.py) | 新增 | `QuerySuggester`：`__init__(deepseek_client)` + `_build_payload` + `_extract_content` + `_parse_completion` + `suggest` / `suggest_async` |
| [easysearch/engine.py](file:///d:/EasySearch/easysearch/engine.py) | 修改 | import + `__init__` 实例化 `self.query_suggester`（L106 后）+ 新增 `suggest_query` / `suggest_query_async`（放 `spell_suggest` 附近） |
| [easysearch/__init__.py](file:///d:/EasySearch/easysearch/__init__.py) | 修改 | import `QuerySuggester` + `__all__` 加导出（仿 `GuideGenerator`） |
| [api/schemas.py](file:///d:/EasySearch/api/schemas.py) | 修改 | 新增 `SuggestResponse(completion: str = "", source: str = "none")`（放 `DropdownResponse` 后） |
| [api/main.py](file:///d:/EasySearch/api/main.py) | 修改 | import `SuggestResponse` + 新增 `GET /api/search/suggest`（异步端点，`partial: Query(min_length=1, max_length=100)`，放 `/api/dropdown` 后） |
| [frontend/index.html](file:///d:/EasySearch/frontend/index.html) | 修改 | `<input id="query">` 包入 `.query-wrap`，新增 `<span id="suggest-ghost" class="suggest-ghost" aria-hidden="true">` |
| [frontend/app.js](file:///d:/EasySearch/frontend/app.js) | 修改 | 状态变量 + `scheduleSuggest`/`requestSuggestion`/`acceptSuggestion`/`clearSuggestion` + input/composition/keydown/blur 事件 |
| [frontend/styles.css](file:///d:/EasySearch/frontend/styles.css) | 修改 | `.search-box input` 拆为 `.query-wrap` + `.query-wrap input`（透明背景 + z-index:2）+ 新增 `.suggest-ghost`（绝对定位 + pointer-events:none + 灰色 + 字体对齐） |
| [tests/test_suggest.py](file:///d:/EasySearch/tests/test_suggest.py) | 新增 | 4 个 TestCase：SuggestParse / SuggestEnabled / EngineSuggest / SuggestEndpoint |

## 复用的现有工具（不重写）

- [easysearch/guide.py](file:///d:/EasySearch/easysearch/guide.py) — `GuideGenerator` 作为 LLM 调用模板（payload 结构、`_extract_content`、`except RuntimeError: return None` 降级模式）
- [easysearch/safety.py](file:///d:/EasySearch/easysearch/safety.py) — `sanitize_for_prompt`（输入清洗）/ `sanitize_text`（输出清洗）
- [easysearch/utils.py](file:///d:/EasySearch/easysearch/utils.py) — `extract_json`（鲁棒 JSON 解析，兼容裸字符串回退）
- [easysearch/dashscope.py](file:///d:/EasySearch/easysearch/dashscope.py) — `DashScopeClient.post_json` / `post_json_async`（M12 重试已内置）
- engine 的 `homepage_dropdown` 历史聚合写法（`recent_queries` + `recent_clicks` + `if sid in self.services` 守卫）

## 关键代码结构

### `QuerySuggester._build_payload`
```
prompt = (
    "你是搜索框补全助手。基于用户已输入的前缀 + 历史，生成一个补全建议。"
    "建议必须严格以用户前缀开头（区分大小写），且长度大于前缀本身。"
    '仅输出JSON：{"completion":"..."}，不要解释/Markdown/引号包裹。'
    f"用户前缀：{safe_partial}\n"
    f"最近搜索：{json.dumps(safe_qs, ensure_ascii=False)}\n"
    f"最近点击服务：{json.dumps(safe_cs, ensure_ascii=False)}"
)
return {"model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "thinking": {"type": "disabled"},
        "stream": False}
```

### `QuerySuggester._parse_completion`（前缀强制）
- `extract_json(raw)` → dict 取 `completion`，或裸字符串回退（剥外层引号/代码块）
- `sanitize_text(completion).strip()`
- **硬性校验**：`completion.startswith(partial)` 且 `len(completion) > len(partial)` 失败 → None
- 限长 50 字符，截断后再校验前缀

### `engine.suggest_query_async`
```
try:
    recent_queries = self.store.recent_queries(user_id, 5)
    recent_click_ids = self.store.recent_clicks(user_id, 5)
    recent_clicked_names = [self.services[sid].service_name
                            for sid in recent_click_ids if sid in self.services]
    return await self.query_suggester.suggest_async(partial, recent_queries, recent_clicked_names)
except Exception:
    logger.warning("suggest_query failed ...", exc_info=True)
    return None
```

### 前端 Tab 补全核心
```js
$("query").addEventListener("keydown", (e) => {
  if (e.key === "Tab") {
    if (acceptSuggestion()) e.preventDefault();  // 有建议才阻止焦点跳转
    return;
  }
  if (e.key === "Enter") { clearSuggestion(); doSearch(); }
});
```

## 验证方法

> ⚠️ 本环境 Shell 不可用（Windows 用户名含撇号致 PowerShell 解析失败），以下命令需用户在外部终端执行。

1. **单元测试**：
   ```
   python -m pytest tests/test_suggest.py -v
   python -m pytest tests/ -v          # 回归全量
   python verify.py                     # 兼容性验证
   ```
2. **手动 E2E**：
   - 启动服务：`python -m uvicorn api.main:app --reload`（或项目既有启动方式）
   - 浏览器打开 `http://localhost:8000`
   - 先搜索几次 + 点击几个服务，建立历史
   - 在搜索框输入"开户"，观察灰色建议出现（需配置 `DEEPSEEK_API_KEY`）
   - 按 Tab，建议应填入 query 框
   - 继续输入，旧建议应立即清除并重新请求
   - 中文输入法合成期不应触发请求
   - 无 API Key 时建议应静默隐藏，不影响正常搜索

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| LLM 延迟（1-3s）远超输入节奏 | 200ms 防抖 + `suggestRequestId` 丢弃过期响应 + 渲染前二次校验当前输入 |
| 建议闪烁 | `input` 事件立即 `clearSuggestion()` 再 schedule |
| 并发请求竞态 | `suggestRequestId` 单调计数器，过期响应直接 return |
| IME 合成期误发请求 | `compositionstart` 置标志位 bail，`compositionend` 重新触发 |
| 前缀不匹配（LLM 不听话） | 三重校验：suggester 服务端 / API 端点 / 前端渲染前三处 `startswith` |
| XSS | 服务端 `sanitize_text` + 限长；前端 `textContent`（非 innerHTML） |
| Tab 破坏键盘导航 | 仅有有效建议时 `preventDefault`；无建议时 Tab 走默认 |
| Rate limit 误伤 | M12 `RateLimitMiddleware` 自动作用；200ms 防抖天然限 RPS≤5；遇 429 静默清建议不重试 |
| `thinking: disabled` 字段兼容性 | 运行时验证；若模型不支持可改为省略 `thinking` 键用默认 |
| 向后兼容 | 纯新增端点 + 模块；零修改 `engine.search` / `homepage_dropdown` / 现有签名；verify.py 契约不变 |
