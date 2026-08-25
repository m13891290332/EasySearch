// EasySearch 知识库管理页前端逻辑：导入/导出/版本/回滚/embedding 状态
const API = "/api";
const $ = (id) => document.getElementById(id);

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmtTime(ts) {
  if (!ts) return "—";
  try {
    return new Date(ts * 1000).toLocaleString("zh-CN");
  } catch (e) {
    return String(ts);
  }
}

async function getJson(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(`${resp.status} ${detail}`);
  }
  return resp.json();
}

// ---------- Embedding 状态 ----------
async function loadStatus() {
  try {
    const s = await getJson(`${API}/kb/embedding-status`);
    $("emb-total").textContent = s.total;
    $("emb-embedded").textContent = s.embedded;
    $("emb-in-progress").textContent = s.in_progress ? "是" : "否";
    $("emb-hash").textContent = s.kb_hash ? s.kb_hash.slice(0, 16) + "…" : "—";
    const errWrap = $("emb-error-wrap");
    if (s.last_error) {
      $("emb-error").textContent = s.last_error;
      errWrap.hidden = false;
    } else {
      errWrap.hidden = true;
    }
  } catch (err) {
    console.warn("load status failed", err);
  }
}

// ---------- 导入 ----------
function setupImport() {
  const dz = $("dropzone");
  const input = $("file-input");

  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("drag-over");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag-over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) importFile(file);
  });
  input.addEventListener("change", () => {
    if (input.files[0]) importFile(input.files[0]);
  });
}

async function importFile(file) {
  const progress = $("import-progress");
  const result = $("import-result");
  progress.hidden = false;
  result.hidden = true;
  $("progress-fill").style.width = "30%";
  $("progress-text").textContent = "读取文件…";

  let payload;
  try {
    const text = await file.text();
    payload = JSON.parse(text);
    if (!Array.isArray(payload)) {
      throw new Error("JSON 根必须为服务记录数组");
    }
  } catch (err) {
    progress.hidden = true;
    showImportResult(false, `文件解析失败：${err.message}`);
    return;
  }

  $("progress-fill").style.width = "60%";
  $("progress-text").textContent = "导入并重建索引…";
  try {
    const resp = await fetch(`${API}/kb/import`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || `${resp.status}`);
    }
    $("progress-fill").style.width = "100%";
    $("progress-text").textContent = "完成";
    showImportResult(
      true,
      `导入成功：版本 ${escapeHtml(data.version_id)}，` +
        `服务数 ${data.services_count}，hash ${escapeHtml(data.kb_hash.slice(0, 16))}…`
    );
    await loadStatus();
    await loadVersions();
  } catch (err) {
    progress.hidden = true;
    showImportResult(false, `导入失败：${err.message}`);
  } finally {
    setTimeout(() => {
      progress.hidden = true;
      $("progress-fill").style.width = "0%";
    }, 1500);
  }
}

function showImportResult(ok, msg) {
  const el = $("import-result");
  el.hidden = false;
  el.className = `import-result ${ok ? "ok" : "err"}`;
  el.textContent = msg;
}

// ---------- 导出 ----------
function setupExport() {
  $("export-btn").addEventListener("click", async () => {
    try {
      const resp = await fetch(`${API}/kb/export`);
      if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(`${resp.status} ${detail}`);
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "knowledge_base.json";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert(`导出失败：${err.message}`);
    }
  });
}

// ---------- 版本列表 + 回滚 ----------
async function loadVersions() {
  const body = $("versions-body");
  body.innerHTML = "";
  try {
    const list = await getJson(`${API}/kb/versions`);
    if (!list.length) {
      body.innerHTML =
        '<tr><td colspan="5" class="empty-hint">暂无版本，导入知识库后将自动生成。</td></tr>';
      return;
    }
    for (const v of list) {
      const tr = document.createElement("tr");
      const active = v.active ? '<span class="badge active">active</span>' : "";
      const rollbackBtn = v.active
        ? ""
        : `<button class="btn btn-rollback" data-vid="${escapeHtml(v.version_id)}">回滚</button>`;
      tr.innerHTML = `
        <td><code>${escapeHtml(v.version_id)}</code></td>
        <td><code title="${escapeHtml(v.kb_hash)}">${escapeHtml(v.kb_hash.slice(0, 16))}…</code></td>
        <td>${fmtTime(v.created_at)}</td>
        <td>${active}</td>
        <td>${rollbackBtn}</td>`;
      body.appendChild(tr);
    }
    body.querySelectorAll(".btn-rollback").forEach((btn) => {
      btn.addEventListener("click", () => doRollback(btn.dataset.vid));
    });
  } catch (err) {
    body.innerHTML = `<tr><td colspan="5" class="err">加载失败：${escapeHtml(err.message)}</td></tr>`;
  }
}

async function doRollback(versionId) {
  if (!confirm(`确认回滚到版本 ${versionId}？当前未保存的修改将丢失。`)) return;
  try {
    const resp = await fetch(
      `${API}/kb/rollback?version_id=${encodeURIComponent(versionId)}`,
      { method: "POST" }
    );
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || `${resp.status}`);
    }
    alert(
      `回滚成功：版本 ${data.version_id}，服务数 ${data.services_count}`
    );
    await loadStatus();
    await loadVersions();
  } catch (err) {
    alert(`回滚失败：${err.message}`);
  }
}

// ---------- 初始化 ----------
$("refresh-status-btn").addEventListener("click", loadStatus);
$("refresh-versions-btn").addEventListener("click", loadVersions);
setupImport();
setupExport();
loadStatus();
loadVersions();
