// M14 实时性能大盘——SSE 订阅 /api/metrics/stream，1s 刷新各阶段延迟 / QPS / 降级高亮。
// 与 kb.js 同约定：$ 选择器 + escapeHtml；不依赖外部框架。
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const setText = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  };

  const fmtPct = (v) => (v * 100).toFixed(1) + "%";
  const fmtMs = (v) => (v ? Number(v).toFixed(1) + " ms" : "0 ms");

  // 高亮：错误率>5% → err；缓存<30% → warn；降级>0 → warn；P95>1s → warn
  const applyHighlights = (data) => {
    const errCard = document.getElementById("card-error");
    const cacheCard = document.getElementById("card-cache");
    const degradedCard = document.getElementById("card-degraded");
    const p95Card = document.getElementById("card-p95");

    if (errCard) {
      errCard.classList.toggle("err", data.error_rate > 0.05 && data.total_requests >= 5);
    }
    if (cacheCard) {
      cacheCard.classList.toggle("warn", data.total_requests >= 5 && data.cache_hit_rate < 0.3);
    }
    if (degradedCard) {
      degradedCard.classList.toggle("warn", data.degraded_count > 0);
    }
    if (p95Card) {
      p95Card.classList.toggle("warn", (data.latency_total.p95 || 0) > 1000);
    }
  };

  const renderStages = (stages) => {
    const body = document.getElementById("stage-body");
    if (!body) return;
    const names = Object.keys(stages || {}).sort();
    if (!names.length) {
      body.innerHTML = '<tr><td colspan="4" class="empty-hint">无阶段数据</td></tr>';
      return;
    }
    body.innerHTML = names
      .map((name) => {
        const s = stages[name] || {};
        return `<tr><td>${escapeHtml(name)}</td><td>${fmtMs(s.p50)}</td><td>${fmtMs(s.p95)}</td><td>${fmtMs(s.p99)}</td></tr>`;
      })
      .join("");
  };

  const renderExternal = (external) => {
    const body = document.getElementById("external-body");
    if (!body) return;
    const entries = Object.entries(external || {});
    if (!entries.length) {
      body.innerHTML = '<div class="empty-hint">无外部调用</div>';
      return;
    }
    body.innerHTML = entries
      .map(([svc, s]) => {
        const rate = ((s.success_rate || 0) * 100).toFixed(1) + "%";
        const failClass = s.consecutive_fail > 0 ? "fail" : "";
        return `<div class="dash-external-row"><span>${escapeHtml(svc)}</span><span>成功率 ${rate} · 连续失败 <span class="${failClass}">${s.consecutive_fail}</span> · 共 ${s.total_calls}</span></div>`;
      })
      .join("");
  };

  const handleEvent = (data) => {
    setText("m-qps", Number(data.qps || 0).toFixed(2));
    setText("m-total", `${data.total_requests || 0} 次请求`);

    setText("m-error", fmtPct(data.error_rate || 0));
    setText("m-errors-count", `${Math.round((data.error_rate || 0) * (data.total_requests || 0))} 次错误`);

    setText("m-cache", fmtPct(data.cache_hit_rate || 0));
    const hits = Math.round((data.cache_hit_rate || 0) * (data.total_requests || 0));
    setText("m-hits", `命中 ${hits}`);

    setText("m-degraded", String(data.degraded_count || 0));

    const lat = data.latency_total || {};
    setText("m-p95", fmtMs(lat.p95));
    setText("m-p99", `P99 ${fmtMs(lat.p99)}`);

    setText("m-db", String(data.db_pool_usage || 0));
    setText("m-emb", data.kb_embedding_in_progress ? "embedding 进行中" : "embedding 空闲");

    renderStages(data.latency_stages);
    renderExternal(data.external);
    applyHighlights(data);
  };

  const setStreamStatus = (state, text) => {
    const el = document.getElementById("stream-status");
    if (el) {
      el.classList.remove("live", "stale");
      if (state) el.classList.add(state);
    }
    setText("stream-text", text);
  };

  const fmtClock = (ts) => {
    if (!ts) return "—";
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString("zh-CN", { hour12: false });
  };

  // 简易 HTML 转义（与 kb.js 同实现，避免重复依赖）
  const escapeHtml = (s) =>
    String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");

  // 连接 SSE 流；断线自动重连（EventSource 原生支持，这里加状态提示）
  const connect = () => {
    const url = "/api/metrics/stream?window=60&interval=1";
    let es;
    try {
      es = new EventSource(url);
    } catch (e) {
      setStreamStatus("stale", "浏览器不支持 SSE");
      return;
    }
    es.onopen = () => setStreamStatus("live", "已连接 · 实时刷新中");
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        handleEvent(data);
        setText("last-update", `更新于 ${fmtClock(data.pushed_at || Date.now() / 1000)}`);
      } catch (e) {
        setStreamStatus("stale", "数据解析失败");
      }
    };
    es.onerror = () => {
      setStreamStatus("stale", "连接断开，重连中…");
    };
  };

  // 启动：先拉一次快照，再开 SSE 流
  fetch("/api/metrics/realtime?window=60")
    .then((r) => r.json())
    .then(handleEvent)
    .catch(() => setStreamStatus("stale", "初始快照拉取失败"))
    .finally(connect);
})();
