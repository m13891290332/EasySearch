from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
from collections import Counter
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_queries (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   TEXT NOT NULL,
    query     TEXT NOT NULL,
    ts        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_user_queries_user ON user_queries(user_id, id);

CREATE TABLE IF NOT EXISTS user_clicks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    service_id  TEXT NOT NULL,
    ts          REAL NOT NULL,
    -- M12：deprecated=1 表示该点击发生时服务已下线（KB 中已不存在）；
    -- 仍记录用于行为分析，但 search 混合打分不会再用其 popularity。
    deprecated  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_user_clicks_user ON user_clicks(user_id, id);
CREATE INDEX IF NOT EXISTS idx_user_clicks_svc_ts ON user_clicks(service_id, ts);

CREATE TABLE IF NOT EXISTS global_clicks (
    service_id  TEXT PRIMARY KEY,
    count       INTEGER NOT NULL DEFAULT 0
);

-- M7 长程对话：每轮 (session_id, turn_idx, query, top_ids) 落库，支持撤回
CREATE TABLE IF NOT EXISTS search_sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    turn_idx     INTEGER NOT NULL,
    query        TEXT NOT NULL,
    top_ids_json TEXT NOT NULL,
    ts           REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_sid_turn ON search_sessions(session_id, turn_idx);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON search_sessions(user_id);

-- M13 负反馈：dwell time 落库，快速跳出（dwell < QUICK_BOUNCE_MS）为负样本
CREATE TABLE IF NOT EXISTS service_feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    service_id  TEXT NOT NULL,
    dwell_ms    INTEGER NOT NULL,
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_svc_ts ON service_feedback(service_id, ts);

-- M9 知识库版本管理：每次导入存一条快照元数据，active 标记当前生效版本
CREATE TABLE IF NOT EXISTS kb_versions (
    version_id  TEXT PRIMARY KEY,
    kb_hash     TEXT NOT NULL,
    path        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    active      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_kb_versions_active ON kb_versions(active);
CREATE INDEX IF NOT EXISTS idx_kb_versions_created ON kb_versions(created_at);

-- M11 全链路行为日志：每次搜索一条，user_id 哈希化保护隐私。
-- 用途：无点击 query 聚合（召回优化）、高延迟 query（性能优化）、降级频次（外部健康）。
-- clicked_sid 默认 NULL；后续 record_click 命中时回填本条 clicked_sid。
CREATE TABLE IF NOT EXISTS search_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_hash       TEXT NOT NULL,
    query           TEXT NOT NULL,
    intent          TEXT NOT NULL DEFAULT '',
    sub_queries_json TEXT NOT NULL DEFAULT '[]',
    top_ids_json    TEXT NOT NULL DEFAULT '[]',
    latencies_json  TEXT NOT NULL DEFAULT '{}',
    cache_hit       INTEGER NOT NULL DEFAULT 0,
    degraded        INTEGER NOT NULL DEFAULT 0,
    session_id      TEXT,
    clicked_sid     TEXT,
    ts              REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_search_logs_ts ON search_logs(ts);
CREATE INDEX IF NOT EXISTS idx_search_logs_query ON search_logs(query);
CREATE INDEX IF NOT EXISTS idx_search_logs_user_hash ON search_logs(user_hash);
CREATE INDEX IF NOT EXISTS idx_search_logs_session ON search_logs(session_id);

-- M11 KB 操作日志：import/export/rollback/embedding 等运维动作留痕
CREATE TABLE IF NOT EXISTS kb_op_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    op          TEXT NOT NULL,
    version_id  TEXT,
    kb_hash     TEXT,
    ok          INTEGER NOT NULL DEFAULT 1,
    detail_json TEXT NOT NULL DEFAULT '{}',
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_kb_op_logs_ts ON kb_op_logs(ts);
CREATE INDEX IF NOT EXISTS idx_kb_op_logs_op ON kb_op_logs(op);
"""

# M11：user_id 哈希化盐值（进程级，可由环境变量 EASYSEARCH_USER_SALT 覆盖）。
# 用途：search_logs.user_hash = sha256(user_id + salt)，不存原始 user_id，保护隐私。
_USER_SALT = os.getenv("EASYSEARCH_USER_SALT", "easysearch-default-salt-v1")


class SQLiteStore:
    """用户行为与热门服务的 SQLite 持久化层。

    - user_queries : 每次搜索记录查询词
    - user_clicks  : 每次点击记录服务
    - global_clicks: 服务->总点击数（首页「最热三服务」来源）

    并发安全：写操作加进程内锁；连接 check_same_thread=False。
    uvicorn 默认单 worker 下足够；多 worker 部署需替换为连接池（超出本计划）。
    """

    def __init__(self, db_path: str = "data/easysearch.db") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        # 自动创建父目录，避免 data/ 不存在导致连接失败（":memory:" 除外）
        if db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        # M12：兼容旧库迁移——user_clicks 表若无 deprecated 列则补加
        self._migrate_add_deprecated_column()

    def _migrate_add_deprecated_column(self) -> None:
        """M12：旧库 user_clicks 表可能没有 deprecated 列，按需 ALTER TABLE 补加。

        SQLite 不支持 ``ALTER TABLE ADD COLUMN IF NOT EXISTS``，所以先查 PRAGMA，
        缺列时才执行 ALTER（保持幂等，多次调用安全）。
        """
        with self._lock:
            try:
                cols = self._conn.execute("PRAGMA table_info(user_clicks)").fetchall()
            except sqlite3.Error:
                return  # 表不存在（不应发生，_SCHEMA 已建表）→ 跳过
            col_names = {row["name"] for row in cols} if cols else set()
            if "deprecated" in col_names:
                return
            try:
                self._conn.execute(
                    "ALTER TABLE user_clicks ADD COLUMN deprecated INTEGER NOT NULL DEFAULT 0"
                )
                self._conn.commit()
            except sqlite3.OperationalError:
                # 并发迁移或表只读：忽略，CREATE TABLE 时新库已建列
                pass

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------- 查询历史 ----------
    def append_query(self, user_id: str, query: str, ts: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO user_queries(user_id, query, ts) VALUES(?,?,?)",
                (user_id, query, ts),
            )
            self._conn.commit()

    def recent_queries(self, user_id: str, limit: int = 3) -> list[str]:
        """返回最近 limit 个未重复的搜索词（最近→最旧）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT query FROM user_queries WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, max(limit * 10, 50)),
            ).fetchall()
        seen: set[str] = set()
        result: list[str] = []
        for row in rows:
            q = row["query"]
            if q not in seen:
                seen.add(q)
                result.append(q)
            if len(result) >= limit:
                break
        return result

    def all_queries(self, user_id: str, limit: int = 50) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT query FROM user_queries WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        # 返回时间正序（旧→新），便于 DIN 序列
        return list(reversed([row["query"] for row in rows]))

    def query_count(self, user_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM user_queries WHERE user_id=?",
                (user_id,),
            ).fetchone()
        return int(row["n"]) if row else 0

    # ---------- 点击历史 ----------
    def append_click(
        self, user_id: str, service_id: str, ts: float, deprecated: bool = False
    ) -> None:
        """记录一次点击。deprecated=True 表示点击时服务已下线（M12：不硬 404）。

        - deprecated=False：正常点击，写入 user_clicks + global_clicks 计数 +1
        - deprecated=True：服务已不在 KB，仅写 user_clicks(deprecated=1) 用于行为分析，
          不污染 global_clicks（避免已下线服务持续计入热度榜）。
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO user_clicks(user_id, service_id, ts, deprecated) "
                "VALUES(?,?,?,?)",
                (user_id, service_id, ts, 1 if deprecated else 0),
            )
            if not deprecated:
                # 已下线服务的点击不计入全局热度（避免僵尸服务霸榜）
                self._conn.execute(
                    "INSERT INTO global_clicks(service_id, count) VALUES(?,1) "
                    "ON CONFLICT(service_id) DO UPDATE SET count=count+1",
                    (service_id,),
                )
            self._conn.commit()

    def recent_clicks(self, user_id: str, limit: int = 3) -> list[str]:
        """返回最近 limit 个未重复的点击服务 id（最近→最旧）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT service_id FROM user_clicks WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, max(limit * 10, 50)),
            ).fetchall()
        seen: set[str] = set()
        result: list[str] = []
        for row in rows:
            sid = row["service_id"]
            if sid not in seen:
                seen.add(sid)
                result.append(sid)
            if len(result) >= limit:
                break
        return result

    # ---------- 全局热门 ----------
    def hot_services(self, limit: int = 3) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT service_id FROM global_clicks ORDER BY count DESC, service_id LIMIT ?",
                (limit,),
            ).fetchall()
        return [row["service_id"] for row in rows]

    def global_click_counter(self) -> Counter[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT service_id, count FROM global_clicks"
            ).fetchall()
        return Counter({row["service_id"]: int(row["count"]) for row in rows})

    def popularity_decayed(
        self,
        tau: float = 2592000.0,
        now: float | None = None,
        window_days: int = 90,
    ) -> dict[str, float]:
        """A4：基于 user_clicks.ts 的时间衰减热度。

        对每个 service，累加 exp(-(now-ts)/tau)，近期点击权重高，
        老点击按指数衰减。window_days 限制 SQL 扫描窗口。

        与 hot_services()（raw count）互补：hot_services 用于下拉"最热"，
        popularity_decayed 用于 search 混合打分，避免老服务永久霸榜。
        """
        if now is None:
            now = time.time()
        cutoff = now - window_days * 86400.0
        with self._lock:
            rows = self._conn.execute(
                "SELECT service_id, ts FROM user_clicks WHERE ts >= ?",
                (cutoff,),
            ).fetchall()
        scores: dict[str, float] = {}
        for row in rows:
            sid = row["service_id"]
            delta = max(0.0, now - float(row["ts"]))
            scores[sid] = scores.get(sid, 0.0) + math.exp(-delta / tau)
        return scores

    def all_click_history(self, user_id: str, limit: int = 50) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT service_id FROM user_clicks WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return list(reversed([row["service_id"] for row in rows]))

    # ---------- M13 负反馈（dwell time）----------
    def append_feedback(
        self, user_id: str, service_id: str, dwell_ms: int, ts: float
    ) -> None:
        """记录一次结果停留时长（dwell_ms）。快速跳出（< QUICK_BOUNCE_MS）为负样本。"""
        with self._lock:
            self._conn.execute(
                "INSERT INTO service_feedback(user_id, service_id, dwell_ms, ts) "
                "VALUES(?,?,?,?)",
                (user_id, service_id, int(dwell_ms), ts),
            )
            self._conn.commit()

    def negative_signals(
        self,
        now: float | None = None,
        window_days: int = 90,
        quick_bounce_ms: int = 3000,
    ) -> dict[str, int]:
        """返回 {service_id: 负样本计数}（window_days 内 dwell < quick_bounce_ms 的次数）。

        用于在 popularity 中扣除快速跳出服务的权重，实现「点后快速跳出降权」。
        """
        if now is None:
            now = time.time()
        cutoff = now - window_days * 86400.0
        with self._lock:
            rows = self._conn.execute(
                "SELECT service_id, COUNT(*) AS n FROM service_feedback "
                "WHERE ts >= ? AND dwell_ms < ? GROUP BY service_id",
                (cutoff, quick_bounce_ms),
            ).fetchall()
        return {row["service_id"]: int(row["n"]) for row in rows}

    # ---------- M7 长程对话会话 ----------
    def session_exists(self, session_id: str) -> bool:
        """会话是否存在（至少一轮）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM search_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        return int(row["n"]) > 0 if row else False

    def session_last_turn_idx(self, session_id: str) -> int:
        """返回会话最大 turn_idx；无轮次返回 -1。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(turn_idx) AS m FROM search_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if row is None or row["m"] is None:
            return -1
        return int(row["m"])

    def session_turns(self, session_id: str) -> list[sqlite3.Row]:
        """返回会话全部轮次，按 turn_idx 升序（旧→新）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT turn_idx, query, top_ids_json, ts FROM search_sessions "
                "WHERE session_id=? ORDER BY turn_idx ASC",
                (session_id,),
            ).fetchall()
        return list(rows)

    def append_session_turn(
        self,
        session_id: str,
        user_id: str,
        turn_idx: int,
        query: str,
        top_ids: list[str],
        ts: float,
    ) -> None:
        """追加一轮会话（query + Top-40 候选 id 列表，JSON 落库）。"""
        with self._lock:
            self._conn.execute(
                "INSERT INTO search_sessions(session_id, user_id, turn_idx, query, top_ids_json, ts) "
                "VALUES(?,?,?,?,?,?)",
                (session_id, user_id, turn_idx, query, json.dumps(top_ids, ensure_ascii=False), ts),
            )
            self._conn.commit()

    def session_delete_last_turn(self, session_id: str) -> bool:
        """弹出末轮（max turn_idx）。返回是否实际删除了一轮。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM search_sessions WHERE session_id=? "
                "ORDER BY turn_idx DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if row is None:
                return False
            self._conn.execute(
                "DELETE FROM search_sessions WHERE id=?",
                (row["id"],),
            )
            self._conn.commit()
            return True

    # ---------- 运维 ----------
    def reset(self) -> None:
        with self._lock:
            self._conn.executescript(
                "DELETE FROM user_queries; DELETE FROM user_clicks; DELETE FROM global_clicks; "
                "DELETE FROM search_sessions; DELETE FROM service_feedback; "
                "DELETE FROM kb_versions; DELETE FROM search_logs; DELETE FROM kb_op_logs;"
            )
            self._conn.commit()

    # ---------- M9 知识库版本元数据 ----------
    def kb_version_add(
        self,
        version_id: str,
        kb_hash: str,
        path: str,
        created_at: float,
        active: bool = True,
    ) -> None:
        """新增一条 KB 版本快照元数据。active=True 时先把其它版本置为非 active 再插入。"""
        with self._lock:
            if active:
                self._conn.execute("UPDATE kb_versions SET active=0")
            self._conn.execute(
                "INSERT OR REPLACE INTO kb_versions(version_id, kb_hash, path, created_at, active) "
                "VALUES(?,?,?,?,?)",
                (version_id, kb_hash, path, created_at, 1 if active else 0),
            )
            self._conn.commit()

    def kb_version_list(self) -> list[dict[str, Any]]:
        """列出全部 KB 版本（新→旧），每项 {version_id, kb_hash, path, created_at, active}。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT version_id, kb_hash, path, created_at, active "
                "FROM kb_versions ORDER BY created_at DESC, version_id DESC"
            ).fetchall()
        return [
            {
                "version_id": row["version_id"],
                "kb_hash": row["kb_hash"],
                "path": row["path"],
                "created_at": float(row["created_at"]),
                "active": bool(row["active"]),
            }
            for row in rows
        ]

    def kb_version_get(self, version_id: str) -> dict[str, Any] | None:
        """单条版本元数据；不存在返回 None。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT version_id, kb_hash, path, created_at, active "
                "FROM kb_versions WHERE version_id=?",
                (version_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "version_id": row["version_id"],
            "kb_hash": row["kb_hash"],
            "path": row["path"],
            "created_at": float(row["created_at"]),
            "active": bool(row["active"]),
        }

    def kb_version_set_active(self, version_id: str) -> bool:
        """把指定版本置为当前 active（其余清零）。版本不存在返回 False。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM kb_versions WHERE version_id=?", (version_id,)
            ).fetchone()
            if row is None:
                return False
            self._conn.execute("UPDATE kb_versions SET active=0")
            self._conn.execute(
                "UPDATE kb_versions SET active=1 WHERE version_id=?", (version_id,)
            )
            self._conn.commit()
        return True

    def kb_version_active(self) -> dict[str, Any] | None:
        """当前 active 版本元数据；无 active 返回 None。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT version_id, kb_hash, path, created_at, active "
                "FROM kb_versions WHERE active=1 LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {
            "version_id": row["version_id"],
            "kb_hash": row["kb_hash"],
            "path": row["path"],
            "created_at": float(row["created_at"]),
            "active": bool(row["active"]),
        }

    # ---------- M11 数据日志 ----------
    @staticmethod
    def hash_user_id(user_id: str) -> str:
        """user_id 哈希化：sha256(user_id + 盐)。

        search_logs 只存哈希，不存原始 user_id，保护隐私；同进程内同 user_id 哈希稳定，
        支持按用户聚合统计（无点击率/高频 query）。盐值由 EASYSEARCH_USER_SALT 覆盖。
        """
        return hashlib.sha256((user_id + _USER_SALT).encode("utf-8")).hexdigest()

    def append_search_log(
        self,
        user_id: str,
        query: str,
        intent: str,
        top_ids: list[str],
        latencies: dict[str, float],
        cache_hit: bool,
        degraded: bool,
        ts: float,
        sub_queries: list[str] | None = None,
        session_id: str | None = None,
    ) -> int:
        """M11：落一条 search_logs 记录，返回自增 id（供 record_click 回填 clicked_sid）。

        - user_id 经 hash_user_id 哈希化后存储
        - top_ids/latencies/sub_queries JSON 序列化
        - cache_hit/degraded 转为 0/1
        - clicked_sid 默认 NULL，后续 mark_search_log_click 回填
        """
        user_hash = self.hash_user_id(user_id)
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO search_logs (
                    user_hash, query, intent, sub_queries_json, top_ids_json,
                    latencies_json, cache_hit, degraded, session_id, clicked_sid, ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    user_hash, query, intent,
                    json.dumps(sub_queries or [], ensure_ascii=False),
                    json.dumps(top_ids, ensure_ascii=False),
                    json.dumps(latencies, ensure_ascii=False),
                    1 if cache_hit else 0,
                    1 if degraded else 0,
                    session_id,
                    ts,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def mark_search_log_click(self, log_id: int, clicked_sid: str) -> bool:
        """M11：回填 search_logs.clicked_sid（用户点击结果时由 engine 调用）。

        返回是否实际更新了一行（log_id 不存在返回 False）。
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE search_logs SET clicked_sid=? WHERE id=? AND clicked_sid IS NULL",
                (clicked_sid, log_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def recent_search_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        """M11：最近 limit 条 search_logs（新→旧），便于调试/巡检。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, user_hash, query, intent, sub_queries_json, top_ids_json, "
                "latencies_json, cache_hit, degraded, session_id, clicked_sid, ts "
                "FROM search_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._search_log_row_to_dict(row) for row in rows]

    @staticmethod
    def _search_log_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "user_hash": row["user_hash"],
            "query": row["query"],
            "intent": row["intent"],
            "sub_queries": json.loads(row["sub_queries_json"]),
            "top_ids": json.loads(row["top_ids_json"]),
            "latencies": json.loads(row["latencies_json"]),
            "cache_hit": bool(row["cache_hit"]),
            "degraded": bool(row["degraded"]),
            "session_id": row["session_id"],
            "clicked_sid": row["clicked_sid"],
            "ts": float(row["ts"]),
        }

    def aggregate_no_click_queries(
        self,
        window_seconds: float = 86400.0,
        now: float | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """M11：按 query 聚合「无点击率」——召回优化的核心信号。

        - 统计窗口（默认 24h）内每个 query 的总搜索次数 + 无点击次数
        - 无点击 = clicked_sid IS NULL（用户搜了但没点任何结果）
        - 返回 {query, total, no_click, no_click_rate} 按 total 降序 limit 条
        - 用途：高频无点击 query → 召回/相关性优化（M13 同义词/负反馈挖掘）
        """
        if now is None:
            now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT query,
                       COUNT(*) AS total,
                       SUM(CASE WHEN clicked_sid IS NULL THEN 1 ELSE 0 END) AS no_click
                FROM search_logs
                WHERE ts >= ? AND ts <= ?
                GROUP BY query
                ORDER BY total DESC, no_click DESC
                LIMIT ?
                """,
                (cutoff, now, limit),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            total = int(row["total"])
            no_click = int(row["no_click"]) if row["no_click"] is not None else 0
            rate = (no_click / total) if total > 0 else 0.0
            result.append({
                "query": row["query"],
                "total": total,
                "no_click": no_click,
                "no_click_rate": round(rate, 4),
            })
        return result

    def aggregate_high_latency_queries(
        self,
        window_seconds: float = 86400.0,
        now: float | None = None,
        latency_threshold_ms: float = 1000.0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """M11：按 query 聚合「高延迟搜索」——性能优化信号。

        - 从 latencies_json.total 提取每次搜索总耗时
        - 统计窗口内每个 query 平均/最大/高延迟次数（超过 latency_threshold_ms）
        - 返回 {query, total, avg_total_ms, max_total_ms, slow_count} 按 slow_count 降序
        - 用途：慢 query → 检索/外部调用性能优化
        """
        if now is None:
            now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT query, latencies_json
                FROM search_logs
                WHERE ts >= ? AND ts <= ?
                """,
                (cutoff, now),
            ).fetchall()
        stats: dict[str, dict[str, Any]] = {}
        for row in rows:
            q = row["query"]
            try:
                latencies = json.loads(row["latencies_json"])
            except (json.JSONDecodeError, TypeError):
                latencies = {}
            total_ms = float(latencies.get("total", 0.0))
            entry = stats.setdefault(q, {
                "query": q, "total": 0, "total_ms_sum": 0.0,
                "max_total_ms": 0.0, "slow_count": 0,
            })
            entry["total"] += 1
            entry["total_ms_sum"] += total_ms
            if total_ms > entry["max_total_ms"]:
                entry["max_total_ms"] = total_ms
            if total_ms >= latency_threshold_ms:
                entry["slow_count"] += 1
        result: list[dict[str, Any]] = []
        for entry in stats.values():
            total = entry["total"]
            result.append({
                "query": entry["query"],
                "total": total,
                "avg_total_ms": round(entry["total_ms_sum"] / total, 2) if total else 0.0,
                "max_total_ms": round(entry["max_total_ms"], 2),
                "slow_count": entry["slow_count"],
            })
        result.sort(key=lambda x: x["slow_count"], reverse=True)
        return result[:limit]

    def search_log_degradation_stats(
        self,
        window_seconds: float = 3600.0,
        now: float | None = None,
    ) -> dict[str, Any]:
        """M11：窗口内降级/缓存命中/错误频次——外部服务健康信号。

        返回 {window_seconds, total, cache_hit, cache_hit_rate, degraded, degraded_rate}。
        用途：降级频次高 → 外部服务（embed/rerank/reason）健康度下降，触发告警/降级策略。
        """
        if now is None:
            now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN cache_hit=1 THEN 1 ELSE 0 END) AS cache_hit,
                       SUM(CASE WHEN degraded=1 THEN 1 ELSE 0 END) AS degraded
                FROM search_logs WHERE ts >= ? AND ts <= ?
                """,
                (cutoff, now),
            ).fetchone()
        total = int(rows["total"]) if rows and rows["total"] is not None else 0
        cache_hit = int(rows["cache_hit"]) if rows and rows["cache_hit"] is not None else 0
        degraded = int(rows["degraded"]) if rows and rows["degraded"] is not None else 0
        return {
            "window_seconds": window_seconds,
            "total": total,
            "cache_hit": cache_hit,
            "cache_hit_rate": round(cache_hit / total, 4) if total else 0.0,
            "degraded": degraded,
            "degraded_rate": round(degraded / total, 4) if total else 0.0,
        }

    # ---------- M11 KB 操作日志 ----------
    def append_kb_op_log(
        self,
        op: str,
        version_id: str | None = None,
        kb_hash: str | None = None,
        ok: bool = True,
        detail: dict[str, Any] | None = None,
        ts: float | None = None,
    ) -> int:
        """M11：落一条 kb_op_logs 记录（import/export/rollback/embedding 等）。

        返回自增 id。op 为操作类型字符串，detail 为额外上下文（JSON 序列化）。
        """
        if ts is None:
            ts = time.time()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO kb_op_logs(op, version_id, kb_hash, ok, detail_json, ts)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    op, version_id, kb_hash, 1 if ok else 0,
                    json.dumps(detail or {}, ensure_ascii=False), ts,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def list_kb_op_logs(
        self, limit: int = 50, op: str | None = None
    ) -> list[dict[str, Any]]:
        """M11：列出 KB 操作日志（新→旧），可按 op 过滤。"""
        with self._lock:
            if op is not None:
                rows = self._conn.execute(
                    "SELECT id, op, version_id, kb_hash, ok, detail_json, ts "
                    "FROM kb_op_logs WHERE op=? ORDER BY id DESC LIMIT ?",
                    (op, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, op, version_id, kb_hash, ok, detail_json, ts "
                    "FROM kb_op_logs ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "op": row["op"],
                "version_id": row["version_id"],
                "kb_hash": row["kb_hash"],
                "ok": bool(row["ok"]),
                "detail": json.loads(row["detail_json"]),
                "ts": float(row["ts"]),
            }
            for row in rows
        ]

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())
