// EasySearch 前端：独立搜索主页，fetch 调 FastAPI 后端
// 用户ID：URL ?user= 优先，其次 localStorage，默认 u-demo

const API = "/api";
const userId = new URLSearchParams(location.search).get("user")
  || localStorage.getItem("easysearch_user")
  || "u-demo";
localStorage.setItem("easysearch_user", userId);

const $ = (id) => document.getElementById(id);

// 检索模式标签（用于状态栏展示）
const RETRIEVAL_MODE_LABELS = {
  hybrid: "混合检索",
  keyword: "仅关键词",
  semantic: "仅语义",
};

// M12：HTTP 错误码 → 用户友好消息映射；不回显后端 stack/detail，避免泄露内部细节
const HTTP_ERROR_MESSAGES = {
  400: "请求格式错误",
  401: "未授权，请检查 API Key",
  403: "无权访问该资源",
  404: "资源不存在",
  409: "资源状态冲突",
  413: "上传内容过大",
  422: "请求参数校验失败",
  429: "请求过于频繁，请稍后再试",
  500: "服务暂时不可用，请稍后重试",
  502: "后端服务异常",
  503: "服务暂时不可用",
  504: "请求超时",
};

// M12：把 fetch 失败响应映射为友好消息；4xx 可附简短 detail（FastAPI 标准 detail 字段），
// 5xx 仅返回通用化消息，不读取后端响应体（避免泄露内部异常栈）
async function mapHttpError(resp) {
  const friendly = HTTP_ERROR_MESSAGES[resp.status] || `请求失败（${resp.status}）`;
  // 仅 4xx 尝试解析 detail（FastAPI HTTPException 的 detail 字段是用户可见消息）
  if (resp.status >= 400 && resp.status < 500) {
    try {
      const body = await resp.json();
      const detail = body && body.detail ? String(body.detail) : "";
      if (detail) {
        // 截断防止超长 detail 干扰 UI；只取首 200 字符
        const trimmed = detail.length > 200 ? detail.slice(0, 200) + "…" : detail;
        return new Error(`${friendly}：${trimmed}`);
      }
    } catch (_e) {
      // 响应非 JSON 或解析失败 → 仅用友好消息，不泄露原始文本
    }
  }
  return new Error(friendly);
}

async function getJson(url, options) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    throw await mapHttpError(resp);
  }
  return resp.json();
}

// 首页下拉默认项：最近3未重复搜索词 / 最近3未重复点击服务 / 全局最热3服务
async function loadDropdown() {
  try {
    const data = await getJson(`${API}/dropdown?user_id=${encodeURIComponent(userId)}`);
    // 最近搜索：点击只填入搜索框，不触发搜索
    renderChips("recent-queries", data.recent_queries || [], {
      label: (q) => q,
      onClick: (q) => { $("query").value = q; },
    });
    // 最近点击：点击直接进入该服务详情（不重新搜索）
    renderChips("recent-clicks", data.recent_clicked_services || [], {
      label: (item) => item.service_name,
      onClick: (item) => showService(item.service_id),
    });
    // 热门服务：点击直接进入该服务详情（不重新搜索）
    renderChips("hot-services", data.global_hot_services || [], {
      label: (item) => item.service_name,
      onClick: (item) => showService(item.service_id),
    });
    $("dropdown").hidden = !(
      (data.recent_queries || []).length ||
      (data.recent_clicked_services || []).length ||
      (data.global_hot_services || []).length
    );
  } catch (err) {
    console.warn("dropdown failed", err);
  }
}

// 通用 chip 渲染：items 可为字符串数组或对象数组，label/onClick 按类型适配
function renderChips(targetId, items, { label, onClick }) {
  const ul = $(targetId);
  ul.innerHTML = "";
  if (!items.length) {
    const li = document.createElement("li");
    li.className = "empty-hint";
    li.textContent = "暂无";
    ul.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.className = "chip";
    btn.type = "button";
    btn.textContent = label(item);
    btn.addEventListener("click", () => onClick(item));
    li.appendChild(btn);
    ul.appendChild(li);
  });
}

// 搜索（会话模式开启时走 M7 长程对话端点，否则走默认 /api/search）
let sessionActive = false;
let sessionId = null;

// 搜索框灰色补全建议（Chrome omnibox 风格）：防抖请求 + Tab 接受
let suggestTimer = null;
let suggestRequestId = 0;        // 单调递增，丢弃过期响应防闪烁
let currentSuggestion = null;    // { partial, completion } 或 null
let composing = false;           // IME 合成态标记（合成期不发请求）
const SUGGEST_DEBOUNCE_MS = 200;

function clearSuggestion() {
  currentSuggestion = null;
  $("suggest-ghost").textContent = "";
}

function scheduleSuggest() {
  clearTimeout(suggestTimer);
  clearSuggestion();              // 立即清旧建议，防闪烁
  if (composing) return;          // 中文合成中不请求
  const partial = $("query").value.trim();
  if (!partial || partial.length > 100) return;
  suggestTimer = setTimeout(() => requestSuggestion(partial), SUGGEST_DEBOUNCE_MS);
}

async function requestSuggestion(partial) {
  const myId = ++suggestRequestId;
  try {
    const url =
      `${API}/search/suggest?user_id=${encodeURIComponent(userId)}` +
      `&partial=${encodeURIComponent(partial)}`;
    const data = await getJson(url);
    if (myId !== suggestRequestId) return;   // 过期响应丢弃（竞态防护）
    if (
      data.completion &&
      data.completion.startsWith(partial) &&
      data.completion.length > partial.length
    ) {
      // 二次校验当前输入仍为该 partial（用户可能已继续输入）
      if ($("query").value !== partial) {
        clearSuggestion();
        return;
      }
      currentSuggestion = { partial, completion: data.completion };
      $("suggest-ghost").textContent = currentSuggestion.completion;
    } else {
      clearSuggestion();
    }
  } catch (_err) {
    clearSuggestion();            // 静默失败，不打扰用户
  }
}

// Tab 接受建议：成功返回 true（阻止默认焦点跳转），失败返回 false（Tab 走默认）
function acceptSuggestion() {
  if (!currentSuggestion) return false;
  const cur = $("query").value;
  const s = currentSuggestion;
  // 三重校验：partial 一致 + completion 仍以 cur 开头
  if (s.partial === cur && s.completion.startsWith(cur)) {
    $("query").value = s.completion;
    clearSuggestion();
    return true;
  }
  clearSuggestion();
  return false;
}

async function doSearch(query) {
  flushPendingDwell();  // M13：开始新搜索前上报上一条结果的 dwell time
  query = (query || $("query").value).trim();
  if (!query) return;
  $("query").value = query;
  if (sessionActive) {
    await doSessionSearch(query);
    return;
  }
  $("status").textContent = "搜索中…";
  $("results").innerHTML = "";
  const retrievalMode = $("retrieval-mode").value;
  const modeLabel = RETRIEVAL_MODE_LABELS[retrievalMode] || retrievalMode;
  try {
    const data = await getJson(
      `${API}/search?user_id=${encodeURIComponent(userId)}` +
      `&query=${encodeURIComponent(query)}&retrieval_mode=${encodeURIComponent(retrievalMode)}`
    );
    if (data.answer_guide && (data.answer_guide.steps || []).length) {
      renderAnswerGuide(data.answer_guide, query);
      $("status").textContent = `指引答案 · ${data.answer_guide.steps.length} 步`;
    } else {
      renderSpellSuggestion(data.spell_suggestion, query);
      renderResults(data.results || [], false, query);
      $("status").textContent =
        `${modeLabel} · 返回 ${(data.results || []).length} 条结果`;
    }
  } catch (err) {
    $("results").innerHTML = `<div class="error">搜索失败：${err.message}</div>`;
    $("status").textContent = "搜索失败";
  }
  // 搜索后刷新下拉（历史/热门可能变化）
  loadDropdown();
}

// M7 长程对话：首轮宽召回，后续轮基于会话上下文精化
async function doSessionSearch(query) {
  $("status").textContent = "会话搜索中…";
  $("results").innerHTML = "";
  renderSpellSuggestion(null, "");  // 会话模式不展示拼写建议，清掉旧值
  try {
    const data = await postJson(`${API}/search/session`, {
      session_id: sessionId,
      user_id: userId,
      query,
      action: "search",
    });
    renderResults(data.results || [], false, query);
    renderSessionTurns(data.history || [], data.turn_idx);
    $("rollback-btn").disabled = (data.history || []).length <= 1;
    $("status").textContent =
      data.match_mode === "session"
        ? `会话第 ${data.turn_idx + 1} 轮 · 返回 ${data.results.length} 条`
        : `会话搜索：${data.match_mode}`;
  } catch (err) {
    $("results").innerHTML = `<div class="error">会话搜索失败：${err.message}</div>`;
    $("status").textContent = "会话搜索失败";
  }
}

// M7 撤回：弹出末轮，返回上一轮结果与上下文
async function rollbackSession() {
  if (!sessionId) return;
  $("status").textContent = "撤回中…";
  try {
    const data = await postJson(`${API}/search/session`, {
      session_id: sessionId,
      user_id: userId,
      query: "",
      action: "rollback",
    });
    renderResults(data.results || [], false, data.query || "");
    renderSessionTurns(data.history || [], data.turn_idx);
    $("rollback-btn").disabled = (data.history || []).length <= 1;
    $("status").textContent =
      data.match_mode === "rollback"
        ? `已撤回至第 ${data.turn_idx + 1} 轮 · 返回 ${data.results.length} 条`
        : `会话已空`;
  } catch (err) {
    $("status").textContent = `撤回失败：${err.message}`;
  }
}

// 渲染会话轮次 chips：点击只回填搜索框（不重新搜索，便于改写后重发）
function renderSessionTurns(history, currentTurnIdx) {
  const ul = $("session-turns");
  ul.innerHTML = "";
  if (!history.length) {
    const li = document.createElement("li");
    li.className = "empty-hint";
    li.textContent = "暂无轮次";
    ul.appendChild(li);
    return;
  }
  history.forEach((t) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.className = "chip" + (t.turn_idx === currentTurnIdx ? " chip-current" : "");
    btn.type = "button";
    btn.textContent = `#${t.turn_idx + 1} ${t.query}`;
    btn.addEventListener("click", () => { $("query").value = t.query; });
    li.appendChild(btn);
    ul.appendChild(li);
  });
}

// 开启/关闭会话模式：开启时生成一个新 sessionId（不复用旧的，避免脏上下文）
function toggleSession() {
  sessionActive = !sessionActive;
  const btn = $("session-toggle");
  const panel = $("session-panel");
  if (sessionActive) {
    sessionId = "s-" + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
    btn.textContent = "会话模式：开";
    btn.classList.add("active");
    panel.hidden = false;
    renderSessionTurns([], -1);
    $("rollback-btn").disabled = true;
    // 多条件与会话互斥：开启会话时收起高级搜索面板
    const advPanel = $("advanced-panel");
    if (advPanel) {
      advPanel.hidden = true;
      $("advanced-toggle").classList.remove("active");
    }
    $("status").textContent = "会话模式已开启，输入查询开始多轮对话";
  } else {
    sessionId = null;
    btn.textContent = "会话模式：关";
    btn.classList.remove("active");
    panel.hidden = true;
    $("status").textContent = "会话模式已关闭";
  }
}

async function postJson(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    // M12：错误码化展示，不回显后端 detail 文本
    throw await mapHttpError(resp);
  }
  return resp.json();
}

// 直接进入单个服务详情（不经过检索/rerank）
async function showService(serviceId) {
  $("status").textContent = "加载服务详情…";
  $("results").innerHTML = "";
  renderSpellSuggestion(null, "");  // 直接访问不展示拼写建议，清掉旧值
  try {
    const item = await getJson(`${API}/service?service_id=${encodeURIComponent(serviceId)}`);
    renderResults([item], true, "");
    $("status").textContent = `服务：${item.service_name}`;
  } catch (err) {
    $("results").innerHTML = `<div class="error">加载服务失败：${err.message}</div>`;
    $("status").textContent = "加载失败";
  }
  loadDropdown();
}

// M16 渲染步骤化指引答案：步骤文本 + 内嵌服务 chip（点击跳转 route）
function renderAnswerGuide(guide, query) {
  const box = $("results");
  box.innerHTML = "";
  const wrap = document.createElement("article");
  wrap.className = "card answer-guide";
  const head = document.createElement("div");
  head.className = "card-head";
  head.innerHTML =
    '<span class="badge badge-direct">指引答案</span>' +
    `<h2 class="svc-name">操作指引 · ${escapeHtml(guide.query || query)}</h2>`;
  wrap.appendChild(head);
  const ol = document.createElement("ol");
  ol.className = "answer-steps";
  (guide.steps || []).forEach((step) => {
    const li = document.createElement("li");
    li.className = "answer-step";
    li.appendChild(renderStepWithRefs(step.step_text || "", step.services || []));
    ol.appendChild(li);
  });
  wrap.appendChild(ol);
  box.appendChild(wrap);
}

// 把 step_text 中的 [[service_id]] 内联标记替换为可点击服务 chip
function renderStepWithRefs(text, services) {
  const frag = document.createDocumentFragment();
  const svcMap = {};
  services.forEach((s) => { svcMap[s.service_id] = s; });
  const re = /\[\[\s*([^\[\]]+?)\s*\]\]/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) {
      frag.appendChild(document.createTextNode(text.slice(last, m.index)));
    }
    const sid = m[1].trim();
    const svc = svcMap[sid];
    if (svc) {
      if (isSafeRoute(svc.route)) {
        const a = document.createElement("a");
        a.className = "chip chip-service-ref";
        a.href = escapeAttr(svc.route);
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = svc.service_name;
        a.title = `${svc.service_name} · ${svc.route}`;
        a.addEventListener("click", () => recordClick(svc.service_id));
        frag.appendChild(a);
      } else {
        const span = document.createElement("span");
        span.className = "chip chip-service-ref btn-disabled";
        span.textContent = svc.service_name;
        span.title = "该路由被安全策略拦截";
        frag.appendChild(span);
      }
    } else {
      // 未知引用：展示为纯文本（去掉括号），避免出现裸 [[...]]
      frag.appendChild(document.createTextNode(sid));
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) {
    frag.appendChild(document.createTextNode(text.slice(last)));
  }
  return frag;
}

// M13 渲染拼写建议条「您是不是要找：xxx」（点击以建议词重搜）；无建议时隐藏
function renderSpellSuggestion(suggestion, originalQuery) {
  const box = $("spell-suggestion");
  if (!box) return;
  if (!suggestion || suggestion === originalQuery) {
    box.hidden = true;
    box.innerHTML = "";
    return;
  }
  box.hidden = false;
  box.innerHTML =
    '<span class="spell-label">您是不是要找：</span>' +
    `<button class="spell-link" type="button">${escapeHtml(suggestion)}</button>`;
  box.querySelector(".spell-link").addEventListener("click", () => {
    $("query").value = suggestion;
    doSearch(suggestion);
  });
}

// 渲染结果卡片。isDirect=true 表示是下拉点击直接进入的单服务（无 score/rerank_reason）
// currentQuery：当前搜索词，用于高亮（直接访问时为空）
function renderResults(results, isDirect, currentQuery) {
  const box = $("results");
  if (!results.length) {
    box.innerHTML = '<div class="empty">未找到相关服务</div>';
    return;
  }
  box.innerHTML = "";
  results.forEach((item, idx) => {
    const card = document.createElement("article");
    card.className = "card";
    let headExtra;
    if (isDirect) {
      headExtra = '<span class="badge badge-direct">直接访问</span>';
    } else {
      // 综合分（rerank_score）决定排序，与展示顺序一致；召回分（score）作辅助
      const rerank = item.rerank_score ?? item.score ?? 0;
      const recall = item.score ?? 0;
      headExtra =
        `<span class="rank">#${idx + 1}</span>` +
        `<span class="score">综合分 ${rerank.toFixed(4)}</span>` +
        `<span class="badge badge-sub">召回分 ${recall.toFixed(4)}</span>`;
    }
    const reason = isDirect
      ? '<div class="reason">💡 该服务来自下拉直接访问，未经过检索重排。</div>'
      : `<div class="reason">💡 ${escapeHtml(item.rerank_reason || "")}</div>`;
    const q = isDirect ? "" : (currentQuery || "");
    card.innerHTML = `
      <div class="card-main">
        <div class="card-head">
          ${headExtra}
          <h2 class="svc-name">${highlight(item.service_name, q)}</h2>
        </div>
        <div class="aliases">${(item.aliases || []).map((a) => highlight(a, q)).join(" · ")}</div>
        <p class="intro">${highlight(truncate(item.service_intro, 200), q)}</p>
        <div class="actions">
          ${
            isSafeRoute(item.route)
              ? `<a class="btn btn-route" href="${escapeAttr(item.route)}" target="_blank" rel="noopener noreferrer">
                   ${escapeHtml(item.decision_button || "进入")} →
                 </a>`
              : `<span class="btn btn-route btn-disabled" title="该路由被安全策略拦截">🚫 进入 →</span>`
          }
          <span class="badge">组件：${escapeHtml(item.component || "-")}</span>
          <span class="badge">路由：${escapeHtml(item.route || "-")}</span>
        </div>
        ${renderComponentActions(item)}
        ${reason}
      </div>
      <aside class="card-deep" data-sid="${escapeAttr(item.service_id)}"></aside>
    `;
    // 点击记录（仅搜索结果记录点击行为；直接访问也可记录以更新热门）
    const routeBtn = card.querySelector(".btn-route");
    if (routeBtn && !routeBtn.classList.contains("btn-disabled")) {
      routeBtn.addEventListener("click", () => recordClick(item.service_id));
    }
    bindComponentActions(card, item);
    box.appendChild(card);
  });
  // 深度检索：勾选后异步填充每条结果右侧的「最佳组件」chip
  doDeepComponents(results.slice(0, 10));
}

// 深度组件检索：对 top-10 结果调 /api/search/deep-components，把最契合 query
// 的组件 chip 渲染到对应 .card-deep（按 data-sid 匹配，避免 CSS 选择器转义问题）。
// chip 点击：有 component+action → 执行组件动作；仅有 href → 安全跳转。
async function doDeepComponents(results) {
  const deepToggle = $("deep-toggle");
  if (!deepToggle || !deepToggle.checked || !results || !results.length) return;
  const serviceIds = results.map((r) => r.service_id).filter(Boolean);
  if (!serviceIds.length) return;
  const query = ($("query") && $("query").value) || "";
  try {
    const data = await postJson(`${API}/search/deep-components`, {
      user_id: userId,
      query,
      service_ids: serviceIds,
    });
    const items = (data && data.items) || [];
    document.querySelectorAll(".card-deep").forEach((aside) => {
      const sid = aside.dataset.sid;
      if (!sid) return;
      const item = items.find((it) => it.service_id === sid);
      if (!item) return;
      const chip = document.createElement("div");
      chip.className = "deep-chip";
      chip.title = item.reason || "";
      chip.innerHTML =
        `<span class="deep-label">${escapeHtml(item.label || "")}</span>` +
        `<span class="badge badge-sub">${escapeHtml(item.source || "")}</span>`;
      chip.addEventListener("click", () => {
        if (item.component && item.action) {
          // 复用 M8 组件动作执行链路（/api/action/execute 打桩）
          executeDeepComponent(item.service_id, item.component, item.action, aside);
        } else if (item.href && isSafeRoute(item.href)) {
          window.open(item.href, "_blank", "noopener,noreferrer");
        }
      });
      aside.appendChild(chip);
    });
  } catch (err) {
    console.error("Deep components failed", err);
  }
}

// 深度组件 chip 的点击执行：在 chip 下方就地展示执行结果（复用 M8 渲染逻辑）
async function executeDeepComponent(serviceId, component, action, aside) {
  if (!serviceId || !component || !action) return;
  let result = aside.querySelector(".deep-result");
  if (!result) {
    result = document.createElement("div");
    result.className = "deep-result";
    aside.appendChild(result);
  }
  result.innerHTML = '<span class="comp-pending">执行中…</span>';
  try {
    const resp = await executeAction(serviceId, component, action);
    result.innerHTML = renderActionResult(resp, component, action);
  } catch (err) {
    result.innerHTML = `<span class="comp-error">动作失败：${escapeHtml(err.message)}</span>`;
  }
}

// M8 渲染页面内组件动作按钮区：每个 component 一颗按钮
function renderComponentActions(item) {
  const comps = item.components || [];
  if (!comps.length) return "";
  const buttons = comps
    .map(
      (c) =>
        `<button class="btn btn-comp" type="button" data-comp="${escapeAttr(c.name)}" data-action="${escapeAttr(c.action)}" title="执行组件动作：${escapeAttr(c.name)} / ${escapeAttr(c.action)}">${escapeHtml(c.name)}</button>`
    )
    .join("");
  return `<div class="comp-actions"><span class="comp-label">页面内动作：</span>${buttons}</div>`;
}

// 绑定组件按钮点击 → 调 /api/action/execute 并就地展示结果
function bindComponentActions(card, item) {
  card.querySelectorAll(".btn-comp").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const component = btn.dataset.comp;
      const action = btn.dataset.action;
      if (!item.service_id || !component || !action) return;
      btn.disabled = true;
      btn.classList.add("btn-loading");
      let result = card.querySelector(".comp-result");
      if (!result) {
        result = document.createElement("div");
        result.className = "comp-result";
        card.querySelector(".comp-actions").after(result);
      }
      result.innerHTML = '<span class="comp-pending">执行中…</span>';
      try {
        const resp = await executeAction(item.service_id, component, action);
        result.innerHTML = renderActionResult(resp, component, action);
      } catch (err) {
        result.innerHTML = `<span class="comp-error">动作失败：${escapeHtml(err.message)}</span>`;
      } finally {
        btn.disabled = false;
        btn.classList.remove("btn-loading");
      }
    });
  });
}

// M8 调 /api/action/execute 打桩端点
async function executeAction(serviceId, component, action, params) {
  const body = { user_id: userId, service_id: serviceId, component, action };
  if (params) body.params = params;
  return postJson(`${API}/action/execute`, body);
}

function renderActionResult(resp, component, action) {
  if (!resp || !resp.ok) {
    return `<span class="comp-error">动作未完成</span>`;
  }
  const echo = resp.echo || {};
  const params = echo.params ? `<code>${escapeHtml(JSON.stringify(echo.params))}</code>` : "";
  return (
    '<span class="comp-ok">✓ 已执行（打桩）</span>' +
    `<span class="comp-meta">${escapeHtml(component)} / ${escapeHtml(action)}</span>` +
    params
  );
}

// 高亮 query 命中词：先 escape 文本，再用 <mark> 包裹 query 分词
function highlight(text, query) {
  const escaped = escapeHtml(String(text || ""));
  if (!query) return escaped;
  const words = String(query)
    .split(/[\s,，。、;；:：]+/)
    .map((w) => w.trim())
    .filter(Boolean)
    .sort((a, b) => b.length - a.length);
  let result = escaped;
  for (const w of words) {
    const ew = escapeHtml(w);
    if (!ew) continue;
    const re = new RegExp(escapeRegex(ew), "gi");
    result = result.replace(re, (m) => `<mark>${m}</mark>`);
  }
  return result;
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// M13 负反馈：记录结果点击时刻，在下一次搜索/离开时上报 dwell time
let pendingDwell = null;  // { serviceId, clickTs }

function startDwell(serviceId) {
  pendingDwell = { serviceId, clickTs: Date.now() };
}

function flushPendingDwell() {
  if (!pendingDwell) return;
  const { serviceId, clickTs } = pendingDwell;
  pendingDwell = null;
  const dwellMs = Date.now() - clickTs;
  // 离线/异常静默：负反馈是辅助信号，不能阻塞主链路
  try {
    fetch(`${API}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, service_id: serviceId, dwell_ms: dwellMs }),
      keepalive: true,
    }).catch(() => {});
  } catch (err) {
    // ignore
  }
}

async function recordClick(serviceId) {
  startDwell(serviceId);  // M13：开始计时，供下次 flush 上报 dwell
  try {
    await fetch(`${API}/click`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, service_id: serviceId }),
    });
    loadDropdown();
  } catch (err) {
    console.warn("record click failed", err);
  }
}

async function loadHealth() {
  try {
    const h = await getJson(`${API}/health`);
    $("engine-info").textContent = `服务数 ${h.services_count} · DashScope ${
      h.dashscope_enabled ? "已配置" : "离线模式"
    }`;
  } catch (err) {
    $("engine-info").textContent = "后端未连接";
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}
function escapeAttr(s) {
  return escapeHtml(s);
}
function truncate(s, n) {
  s = String(s || "");
  return s.length > n ? s.slice(0, n) + "…" : s;
}

// M1 路由安全守卫：仅允许相对路径或 http(s)://mailto，拒绝 javascript:/data: 等
function isSafeRoute(route) {
  if (!route) return false;
  const r = String(route).trim().toLowerCase();
  if (!r) return false;
  if (/^(javascript|data|vbscript|file|about):/.test(r)) return false;
  const m = r.match(/^([a-z][a-z0-9+.\-]*):(?:\/\/|(?=$))/);
  if (m) return m[1] === "http" || m[1] === "https" || m[1] === "mailto";
  return true; // 无 scheme：相对路径
}

// ---------- 高级多条件搜索：+/- 行 + 多条件交集检索（M6） ----------
// 各条件独立召回 Top-30 求交集（空则 RRF 并集兜底）→ qwen3-vl-rerank 重排 + 理由生成
const MC_MIN_ROWS = 2;
const mcRows = () => document.querySelectorAll("#mc-rows .mc-row");

// 重排行号占位 + 最少行数时禁用减号
function refreshMCRows() {
  const rows = mcRows();
  rows.forEach((row, i) => {
    const input = row.querySelector(".mc-input");
    if (input) input.placeholder = `条件 ${i + 1}`;
    const rm = row.querySelector(".mc-rm");
    if (rm) rm.disabled = rows.length <= MC_MIN_ROWS;
  });
}

function makeMCRow(value = "", focus = false) {
  const row = document.createElement("div");
  row.className = "mc-row";
  const input = document.createElement("input");
  input.className = "mc-input";
  input.type = "text";
  input.autocomplete = "off";
  input.placeholder = "条件 N";
  input.value = value;
  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.className = "mc-add";
  addBtn.textContent = "+";
  addBtn.title = "增加一行";
  addBtn.addEventListener("click", () => {
    // 在当前行后插入新空行并聚焦
    const fresh = makeMCRow("", true);
    row.after(fresh);
    refreshMCRows();
  });
  const rmBtn = document.createElement("button");
  rmBtn.type = "button";
  rmBtn.className = "mc-rm";
  rmBtn.textContent = "−";
  rmBtn.title = "删除该行";
  rmBtn.addEventListener("click", () => {
    if (mcRows().length <= MC_MIN_ROWS) return;
    row.remove();
    refreshMCRows();
  });
  row.append(input, addBtn, rmBtn);
  $("mc-rows").appendChild(row);
  refreshMCRows();
  if (focus) input.focus();
  return row;
}

function initMCRows() {
  $("mc-rows").innerHTML = "";
  makeMCRow();
  makeMCRow();
}

function toggleAdvanced() {
  const panel = $("advanced-panel");
  const btn = $("advanced-toggle");
  const open = panel.hidden;
  panel.hidden = !open;
  btn.classList.toggle("active", open);
  // 多条件与会话互斥：开启高级搜索时关闭会话模式
  if (open && sessionActive) toggleSession();
}

async function doMultiConditionSearch() {
  flushPendingDwell();
  const queries = Array.from(mcRows())
    .map((r) => r.querySelector(".mc-input").value.trim())
    .filter(Boolean);
  if (queries.length < 2) {
    $("status").textContent = "至少需要 2 个非空条件";
    return;
  }
  const original = queries.join(" ");
  $("status").textContent = "多条件检索中…";
  $("results").innerHTML = "";
  renderSpellSuggestion(null, "");
  const btn = $("mc-search");
  btn.disabled = true;
  try {
    const data = await postJson(`${API}/search/intersection`, {
      user_id: userId,
      queries,
      original_query: original,
    });
    renderResults(data.results || [], false, original);
    // 顶部加交集/并集命中徽章
    const box = $("results");
    const head = document.createElement("div");
    head.className = "card";
    head.style.padding = "8px 14px";
    const isInter = data.match_mode === "intersection";
    const label = isInter
      ? "交集命中（各条件同时满足）"
      : "并集兜底（无交集，RRF 融合）";
    head.innerHTML =
      `<span class="badge badge-match">${escapeHtml(data.match_mode)}</span> ` +
      `<span class="badge">${escapeHtml(label)}</span>`;
    box.insertBefore(head, box.firstChild);
    renderSpellSuggestion(data.spell_suggestion, original);
    $("status").textContent =
      `多条件 · ${data.match_mode} · ${(data.results || []).length} 条结果`;
  } catch (err) {
    $("results").innerHTML = `<div class="error">多条件搜索失败：${err.message}</div>`;
    $("status").textContent = "多条件搜索失败";
  } finally {
    btn.disabled = false;
  }
  loadDropdown();
}

// 事件绑定
$("search-btn").addEventListener("click", () => doSearch());
$("query").addEventListener("input", scheduleSuggest);
$("query").addEventListener("compositionstart", () => {
  composing = true;
  clearSuggestion();
});
$("query").addEventListener("compositionend", () => {
  composing = false;
  scheduleSuggest();
});
$("query").addEventListener("keydown", (e) => {
  if (e.key === "Tab") {
    // 有有效建议 → 接受并阻止焦点跳转；无建议 → Tab 走默认（移焦点，保可访问性）
    if (acceptSuggestion()) e.preventDefault();
    return;
  }
  if (e.key === "Enter") {
    clearSuggestion();
    doSearch();
  }
});
$("query").addEventListener("focus", () => {
  if (!$("query").value) loadDropdown();
});
$("query").addEventListener("blur", () => {
  // 延迟清，避免与潜在 click 冲突
  setTimeout(() => clearSuggestion(), 150);
});
$("session-toggle").addEventListener("click", () => toggleSession());
$("rollback-btn").addEventListener("click", () => rollbackSession());
$("advanced-toggle").addEventListener("click", () => toggleAdvanced());
$("mc-search").addEventListener("click", () => doMultiConditionSearch());
// M13：页面关闭/刷新时尽力上报最后一条结果的 dwell time（keepalive 保活）
window.addEventListener("pagehide", () => flushPendingDwell());

// 初始化
initMCRows();       // 高级多条件搜索初始 2 行
loadHealth();
loadDropdown();
