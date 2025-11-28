import asyncio
import os
import contextlib
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

try:
    import requests  # Used to pull ads list from Ad Manager
except Exception:
    requests = None

import sqlite3
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


app = FastAPI()
templates = Jinja2Templates(directory="templates")
AD_MANAGER_URL = os.environ.get("AD_MANAGER_URL")  # e.g., http://<jetson-ip>:5002

# Track connected websocket clients for broadcast messages
_ws_clients = set()

ALLOWED_EXTENSIONS = {
    'images': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'},
    'videos': {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}
}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in (ALLOWED_EXTENSIONS['images'] | ALLOWED_EXTENSIONS['videos'])


def get_db():
    conn = sqlite3.connect("camera_analytics.db")
    conn.row_factory = sqlite3.Row
    return conn

def ensure_db_schema():
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ad_plays (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ad_id TEXT NOT NULL,
                start_ts REAL NOT NULL,
                end_ts REAL,
                duration_sec REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                ad_id TEXT,
                processing_time_ms REAL,
                total_persons INTEGER,
                total_faces INTEGER,
                current_tracked_persons INTEGER,
                unique_tracked_persons INTEGER,
                gender_male INTEGER,
                gender_female INTEGER,
                gender_unknown INTEGER,
                new_unique_persons INTEGER,
                new_unique_faces INTEGER,
                new_gender_male INTEGER,
                new_gender_female INTEGER,
                new_gender_unknown INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS presence_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                ad_id TEXT,
                start_ts REAL NOT NULL,
                end_ts REAL NOT NULL,
                duration_sec REAL NOT NULL,
                age REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recent_tracks (
                track_id INTEGER NOT NULL,
                ad_id TEXT,
                last_ts REAL NOT NULL,
                PRIMARY KEY(track_id, ad_id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

# Initialize schema on import
ensure_db_schema()


def parse_window(window: str):
    now = datetime.now(timezone.utc).timestamp()
    if isinstance(window, str) and window.startswith("range:"):
        try:
            _, s, e = window.split(":", 2)
            start_ts = float(s)
            end_ts = float(e)
        except Exception:
            end_ts = now
            start_ts = end_ts - 3600
    elif window.endswith("h"):
        hours = int(window[:-1])
        end_ts = now
        start_ts = end_ts - hours * 3600
    elif window.endswith("d"):
        days = int(window[:-1])
        end_ts = now
        start_ts = end_ts - days * 86400
    else:
        end_ts = now
        start_ts = end_ts - 3600
    span = max(0.0, end_ts - start_ts)
    bucket_seconds = 60 if span <= 12 * 3600 else 300
    return start_ts, end_ts, bucket_seconds


def query_summary(window: str) -> Dict[str, Any]:
    start_ts, end_ts, _ = parse_window(window)

    conn = get_db()
    cur = conn.cursor()
    # Processing and gender stats from analytics; use NEW gender counts (windowed)
    cur.execute(
        """
        SELECT
            COUNT(*) AS samples,
            COALESCE(SUM(new_unique_faces), 0) AS unique_faces_sum,
            MAX(unique_tracked_persons) AS unique_tracked_persons_max,
            COALESCE(SUM(new_gender_male), 0) AS male_sum,
            COALESCE(SUM(new_gender_female), 0) AS female_sum,
            COALESCE(SUM(new_gender_unknown), 0) AS unknown_sum,
            COALESCE(SUM(processing_time_ms), 0) AS sum_processing_ms
        FROM analytics
        WHERE ts >= ? AND ts <= ?
        """,
        (start_ts, end_ts),
    )
    row = cur.fetchone()

    # Footfall should reflect the selected window: count arrivals in the window
    cur.execute(
        """
        SELECT COUNT(*)
        FROM presence_log
        WHERE start_ts >= ? AND start_ts <= ?
        """,
        (start_ts, end_ts),
    )
    footfall_sessions = int((cur.fetchone() or [0])[0] or 0)
    # Fallback: if no presence sessions exist in the window, estimate arrivals from analytics for the window
    if footfall_sessions == 0:
        cur.execute(
            """
            SELECT COALESCE(SUM(new_unique_persons), 0)
            FROM analytics
            WHERE ts >= ? AND ts <= ?
            """,
            (start_ts, end_ts),
        )
        footfall_sessions = int((cur.fetchone() or [0])[0] or 0)
    # Count ad plays in the selected window
    cur.execute(
        """
        SELECT COUNT(*)
        FROM ad_plays
        WHERE end_ts >= ? AND end_ts <= ?
        """,
        (start_ts, end_ts),
    )
    ad_plays_count = int((cur.fetchone() or [0])[0] or 0)
    conn.close()

    samples = int(row[0] or 0)
    sum_ms = float(row[6] or 0.0)
    avg_ms = (sum_ms / samples) if samples > 0 else 0.0
    est_fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0

    return {
        "samples": samples,
        "footfall": int((row[3] or 0) + (row[4] or 0) + (row[5] or 0)),
        "faces": int(row[1] or 0),
        "unique_tracked_persons": int(row[2] or 0),
        "gender": {
            "male": int(row[3] or 0),
            "female": int(row[4] or 0),
            "unknown": int(row[5] or 0),
        },
        "avg_processing_ms": avg_ms,
        "estimated_fps": est_fps,
        "ad_plays": ad_plays_count,
    }


def query_footfall_series(window: str) -> Dict[str, Any]:
    start_ts, end_ts, bucket_seconds = parse_window(window)

    conn = get_db()
    cur = conn.cursor()
    rows: list = []
    try:
        # Prefer presence starts for arrivals per bucket
        cur.execute(
            """
            SELECT CAST(start_ts / ? AS INTEGER) * ? AS bucket, COUNT(*) AS cnt
            FROM presence_log
            WHERE start_ts >= ? AND start_ts <= ?
            GROUP BY bucket
            ORDER BY bucket
            """,
            (bucket_seconds, bucket_seconds, start_ts, end_ts),
        )
        rows = cur.fetchall()
    except Exception:
        rows = []
    # Fallback: if presence is sparse, use analytics new_unique_persons per bucket
    if not rows:
        cur.execute(
            """
            SELECT CAST(ts / ? AS INTEGER) * ? AS bucket, COALESCE(SUM(new_unique_persons), 0) AS cnt
            FROM analytics
            WHERE ts >= ? AND ts <= ?
            GROUP BY bucket
            ORDER BY bucket
            """,
            (bucket_seconds, bucket_seconds, start_ts, end_ts),
        )
        rows = cur.fetchall()
    conn.close()

    # Convert per-bucket counts into a cumulative series that never decreases
    cum = 0
    series = []
    for (b, c) in rows:
        cnt = int(c or 0)
        cum += cnt
        series.append({"t": int(b), "count": cum})
    return {"bucketSeconds": bucket_seconds, "series": series}


def query_gender_series(window: str) -> Dict[str, Any]:
    start_ts, end_ts, bucket_seconds = parse_window(window)

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT CAST(ts / ? AS INTEGER) * ? AS bucket,
               COALESCE(SUM(new_gender_male), 0) AS m,
               COALESCE(SUM(new_gender_female), 0) AS f,
               COALESCE(SUM(new_gender_unknown), 0) AS u
        FROM analytics
        WHERE ts >= ? AND ts <= ?
        GROUP BY bucket
        ORDER BY bucket
        """,
        (bucket_seconds, bucket_seconds, start_ts, end_ts),
    )
    rows = cur.fetchall()
    conn.close()

    series = [
        {"t": int(b), "male": int(m or 0), "female": int(f or 0), "unknown": int(u or 0)}
        for (b, m, f, u) in rows
    ]
    return {"bucketSeconds": bucket_seconds, "series": series}


def query_age_summary(window: str) -> Dict[str, Any]:
    start_ts, end_ts, _ = parse_window(window)

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
          SUM(CASE WHEN age BETWEEN 0 AND 15 THEN 1 ELSE 0 END) AS child,
          SUM(CASE WHEN age BETWEEN 16 AND 40 THEN 1 ELSE 0 END) AS young_adult,
          SUM(CASE WHEN age > 40 THEN 1 ELSE 0 END) AS adult,
          SUM(CASE WHEN age IS NULL OR age < 0 THEN 1 ELSE 0 END) AS unknown
        FROM presence_log
        WHERE end_ts >= ? AND end_ts <= ?
        """,
        (start_ts, end_ts),
    )
    row = cur.fetchone()
    conn.close()
    return {
        "child": int((row or [0, 0, 0, 0])[0] or 0),
        "young_adult": int((row or [0, 0, 0, 0])[1] or 0),
        "adult": int((row or [0, 0, 0, 0])[2] or 0),
        "unknown": int((row or [0, 0, 0, 0])[3] or 0),
    }


def query_presence_summary(window: str) -> Dict[str, Any]:
    start_ts, end_ts, _ = parse_window(window)

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(SUM(duration_sec), 0.0) AS total_sec,
               COUNT(*) AS sessions
        FROM presence_log
        WHERE end_ts >= ? AND end_ts <= ?
        """,
        (start_ts, end_ts),
    )
    row = cur.fetchone()

    cur.execute(
        """
        SELECT track_id, COALESCE(SUM(duration_sec), 0.0) AS total_sec
        FROM presence_log
        WHERE end_ts >= ? AND end_ts <= ?
        GROUP BY track_id
        ORDER BY total_sec DESC
        LIMIT 5
        """,
        (start_ts, end_ts),
    )
    top = [{"track_id": int(r[0]), "total_sec": float(r[1])} for r in cur.fetchall()]
    conn.close()

    return {
        "total_presence_sec": float(row[0] or 0.0),
        "sessions": int(row[1] or 0),
        "top_track_ids": top,
    }


def query_presence_stats(window: str) -> Dict[str, Any]:
    start_ts, end_ts, _ = parse_window(window)

    conn = get_db()
    cur = conn.cursor()

    # Histogram bins
    cur.execute(
        """
        SELECT
          SUM(CASE WHEN duration_sec < 5 THEN 1 ELSE 0 END),
          SUM(CASE WHEN duration_sec >= 5 AND duration_sec < 15 THEN 1 ELSE 0 END),
          SUM(CASE WHEN duration_sec >= 15 AND duration_sec < 30 THEN 1 ELSE 0 END),
          SUM(CASE WHEN duration_sec >= 30 AND duration_sec < 60 THEN 1 ELSE 0 END),
          SUM(CASE WHEN duration_sec >= 60 AND duration_sec < 120 THEN 1 ELSE 0 END),
          SUM(CASE WHEN duration_sec >= 120 THEN 1 ELSE 0 END)
        FROM presence_log
        WHERE start_ts >= ? AND start_ts <= ?
        """,
        (start_ts, end_ts),
    )
    bins_row = cur.fetchone()
    bins = [int(b or 0) for b in bins_row] if bins_row else [0, 0, 0, 0, 0, 0]

    # Hour-of-day average dwell time
    cur.execute(
        """
        SELECT CAST(STRFTIME('%H', datetime(start_ts, 'unixepoch', '+5 hours')) AS INTEGER) AS hour,
               AVG(duration_sec) AS avg_sec
        FROM presence_log
        WHERE start_ts >= ? AND start_ts <= ?
        GROUP BY hour
        ORDER BY hour
        """,
        (start_ts, end_ts),
    )
    hour_rows = cur.fetchall()
    by_hour = [{"hour": int(h), "avg_sec": float(a or 0.0)} for (h, a) in hour_rows]

    # Rolling daily mean and 7-day moving average over the last 30 days
    cur.execute(
        """
        SELECT DATE(datetime(start_ts, 'unixepoch')) AS d,
               AVG(duration_sec) AS avg_sec
        FROM presence_log
        WHERE start_ts >= ? AND start_ts <= ?
        GROUP BY d
        ORDER BY d
        """,
        (start_ts, end_ts),
    )
    daily_rows = cur.fetchall()
    conn.close()

    daily = [{"date": d, "avg_sec": float(a or 0.0)} for (d, a) in daily_rows]

    # Compute MA7 in Python
    ma7 = []
    window_vals = []
    for i, item in enumerate(daily):
        window_vals.append(item["avg_sec"])
        if len(window_vals) > 7:
            window_vals.pop(0)
        ma7.append({"date": item["date"], "avg_sec": sum(window_vals) / len(window_vals) if window_vals else 0.0})

    return {
        "histogram": {
            "bins": bins,
            "labels": ["0-5s", "5-15s", "15-30s", "30-60s", "60-120s", ">120s"],
        },
        "by_hour": by_hour,
        "rolling": {
            "daily": daily,
            "ma7": ma7,
        },
    }

def query_ad_stats(window: str) -> Dict[str, Any]:
    start_ts, end_ts, _ = parse_window(window)
    # Sliding window for viewer counts (distinct tracks recently seen)
    viewer_window_sec = 30.0

    conn = get_db()
    cur = conn.cursor()
    rows = []
    play_map = {}
    try:
        # Saved totals per ad for selected window: sum of new_gender_* since window start
        cur.execute(
            """
            SELECT ad_id,
                   COALESCE(SUM(new_gender_male), 0) AS male,
                   COALESCE(SUM(new_gender_female), 0) AS female,
                   COALESCE(SUM(new_gender_unknown), 0) AS unknown
            FROM analytics
            WHERE ts >= ? AND ts <= ? AND ad_id IS NOT NULL AND ad_id <> ''
            GROUP BY ad_id
            """,
            (start_ts, end_ts)
        )
        rows = cur.fetchall()
    except Exception:
        rows = []
    try:
        cur.execute(
            """
            SELECT ad_id,
                   COUNT(*) AS plays,
                   COALESCE(SUM(duration_sec), 0.0) AS total_sec
            FROM ad_plays
            WHERE end_ts >= ? AND end_ts <= ?
            GROUP BY ad_id
            """,
            (start_ts, end_ts),
        )
        play_map = {r[0]: {"plays": int(r[1] or 0), "total_sec": float(r[2] or 0.0)} for r in cur.fetchall()}
    except Exception:
        play_map = {}

    conn.close()

    stats = []
    for r in rows:
        ad_id = r[0]
        male = int(r[1] or 0)
        female = int(r[2] or 0)
        unknown = int(r[3] or 0)
        stats.append({
            "ad_id": ad_id,
            "viewers": male + female + unknown,
            "male": male,
            "female": female,
            "unknown": unknown,
            "plays": play_map.get(ad_id, {}).get("plays", 0),
            "total_sec": play_map.get(ad_id, {}).get("total_sec", 0.0),
        })
    # Fallback: include analytics-only ads (no presence yet) with viewers estimated from new_unique_persons
    try:
        cur.execute(
            """
            SELECT ad_id, COALESCE(SUM(new_unique_persons), 0) AS viewers
            FROM analytics
            WHERE ts >= ? AND ts <= ? AND ad_id IS NOT NULL AND ad_id <> ''
            GROUP BY ad_id
            """,
            (start_ts, end_ts)
        )
        est = {r[0]: int(r[1] or 0) for r in cur.fetchall()}
        # Merge estimates where viewers are 0
        for s in stats:
            if s["viewers"] == 0:
                s["viewers"] = est.get(s["ad_id"], 0)
    except Exception:
        pass
    # Ensure newly uploaded ads appear even without analytics yet
    try:
        files = []
        # Prefer pulling from Ad Manager if configured
        if AD_MANAGER_URL and requests:
            with contextlib.suppress(Exception):
                resp = requests.get(f"{AD_MANAGER_URL}/api/ads", timeout=2.0)
                j = resp.json() if resp and resp.ok else {}
                if j.get("success"):
                    files = [a.get("filename") for a in j.get("ads", []) or [] if a.get("filename")]
        # Fallback to local advertisement folder
        if not files:
            ads_dir = 'advertisement'
            if os.path.isdir(ads_dir):
                files = sorted([f for f in os.listdir(ads_dir) if allowed_file(f)])
        have = set([s["ad_id"] for s in stats])
        for f in files:
            if f not in have and allowed_file(f):
                stats.append({
                    "ad_id": f,
                    "viewers": 0,
                    "current_viewers": 0,
                    "window_viewers": 0,
                    "male": 0,
                    "female": 0,
                    "unknown": 0,
                    "plays": 0,
                    "total_sec": 0.0,
                })
        # Sort stats by filename for stable UI ordering
        stats = sorted(stats, key=lambda x: str(x.get("ad_id") or ""))
    except Exception:
        pass
    return {"stats": stats}

def query_current_ad() -> Dict[str, Any]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ad_id, start_ts FROM ad_plays
        WHERE end_ts IS NULL
        ORDER BY start_ts DESC
        LIMIT 1
        """
    )
    r = cur.fetchone()
    conn.close()
    if not r:
        return {"ad_id": None}
    return {"ad_id": r[0]}


# Removed dummy data seeding and clearing endpoints per request


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    window = "1h"
    try:
        # Initial push
        payload = {
            "type": "snapshot",
            "summary": query_summary(window),
            "footfall": query_footfall_series(window),
            "gender": query_gender_series(window),
            "age": query_age_summary(window),
            "presence": query_presence_summary(window),
            "presence_stats": query_presence_stats(window),
            "window": window,
        }
        print(f"[WS] send snapshot window={window} summary_samples={payload['summary'].get('samples')} footfall={payload['summary'].get('footfall')}")
        await ws.send_json(payload)

        # Listen for window change from client and periodically push updates
        async def sender():
            while True:
                await asyncio.sleep(2.0)
                upd = {
                    "type": "update",
                    "summary": query_summary(window),
                    "footfall": query_footfall_series(window),
                    "gender": query_gender_series(window),
                    "age": query_age_summary(window),
                    "presence": query_presence_summary(window),
                    "presence_stats": query_presence_stats(window),
                }
                try:
                    print(f"[WS] send update window={window} summary_samples={upd['summary'].get('samples')} footfall={upd['summary'].get('footfall')}")
                except Exception:
                    pass
                await ws.send_json(upd)

        sender_task = asyncio.create_task(sender())
        try:
            while True:
                msg = await ws.receive_json()
                if isinstance(msg, dict) and msg.get("type") == "set_window":
                    new_w = str(msg.get("window") or "1h")
                    window = new_w
        finally:
            sender_task.cancel()
            with contextlib.suppress(Exception):
                await sender_task
    except Exception:
        pass
    finally:
        with contextlib.suppress(Exception):
            _ws_clients.discard(ws)

@app.get("/api/ad_stats")
async def api_ad_stats(window: str = "1h"):
    return query_ad_stats(window)

@app.get("/api/current_ad")
async def api_current_ad():
    return query_current_ad()

@app.post("/api/notify_ad_change")
async def api_notify_ad_change():
    # Broadcast a lightweight WS event so clients refresh ads immediately
    dead = []
    for client in list(_ws_clients):
        try:
            await client.send_json({"type": "ad_changed"})
        except Exception:
            dead.append(client)
    for d in dead:
        with contextlib.suppress(Exception):
            _ws_clients.discard(d)
    return {"status": "ok"}
@app.post("/api/ingest")
async def api_ingest(payload: Dict[str, Any]):
    ad_id = payload.get("ad_id")
    analytics = payload.get("analytics") or {}
    tg = analytics.get("tracked_gender_counts") or {}
    ng = analytics.get("new_gender_counts") or {}
    ts = float(analytics.get("timestamp") or datetime.now(timezone.utc).timestamp())
    conn = get_db()
    try:
        cur = conn.cursor()
        print(f"[API] ingest single ad={ad_id} ts={ts} events={len(analytics.get('presence_events') or [])}")
        cur.execute(
            """
            INSERT INTO analytics (
                ts, ad_id, processing_time_ms, total_persons, total_faces,
                current_tracked_persons, unique_tracked_persons,
                gender_male, gender_female, gender_unknown,
                new_unique_persons, new_unique_faces,
                new_gender_male, new_gender_female, new_gender_unknown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                ad_id,
                float(analytics.get("processing_time_ms") or 0.0),
                int(analytics.get("total_persons") or 0),
                int(analytics.get("total_faces") or 0),
                int(analytics.get("current_tracked_persons") or 0),
                int(analytics.get("unique_tracked_persons") or 0),
                int(tg.get("male") or 0),
                int(tg.get("female") or 0),
                int(tg.get("unknown") or 0),
                int(analytics.get("new_unique_persons") or 0),
                int(analytics.get("new_unique_faces") or 0),
                int(ng.get("male") or 0),
                int(ng.get("female") or 0),
                int(ng.get("unknown") or 0),
            ),
        )
        for ev in analytics.get("presence_events") or []:
            cur.execute(
                """
                INSERT INTO presence_log (track_id, ad_id, start_ts, end_ts, duration_sec, age)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(ev.get("track_id") or 0),
                    ad_id,
                    float(ev.get("start_ts") or 0.0),
                    float(ev.get("end_ts") or 0.0),
                    float(ev.get("duration_sec") or 0.0),
                    float(ev.get("age")) if ev.get("age") is not None else None,
                ),
            )
        conn.commit()
        print("[API] ingest single commit ok")
        return {"status": "ok"}
    finally:
        conn.close()

@app.post("/api/ingest_bulk")
async def api_ingest_bulk(payload: Dict[str, Any]):
    ad_id = payload.get("ad_id")
    items = payload.get("analytics_list") or []
    if not isinstance(items, list):
        return {"status": "error", "message": "analytics_list must be a list"}
    conn = get_db()
    try:
        cur = conn.cursor()
        print(f"[API] ingest bulk ad={ad_id} items={len(items)}")
        for analytics in items:
            tg = (analytics or {}).get("tracked_gender_counts") or {}
            ng = (analytics or {}).get("new_gender_counts") or {}
            ts = float((analytics or {}).get("timestamp") or datetime.now(timezone.utc).timestamp())
            cur.execute(
                """
                INSERT INTO analytics (
                    ts, ad_id, processing_time_ms, total_persons, total_faces,
                    current_tracked_persons, unique_tracked_persons,
                    gender_male, gender_female, gender_unknown,
                    new_unique_persons, new_unique_faces,
                    new_gender_male, new_gender_female, new_gender_unknown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    ad_id,
                    float((analytics or {}).get("processing_time_ms") or 0.0),
                    int((analytics or {}).get("total_persons") or 0),
                    int((analytics or {}).get("total_faces") or 0),
                    int((analytics or {}).get("current_tracked_persons") or 0),
                    int((analytics or {}).get("unique_tracked_persons") or 0),
                    int(tg.get("male") or 0),
                    int(tg.get("female") or 0),
                    int(tg.get("unknown") or 0),
                    int((analytics or {}).get("new_unique_persons") or 0),
                    int((analytics or {}).get("new_unique_faces") or 0),
                    int(ng.get("male") or 0),
                    int(ng.get("female") or 0),
                    int(ng.get("unknown") or 0),
                ),
            )
            for ev in (analytics or {}).get("presence_events") or []:
                cur.execute(
                    """
                    INSERT INTO presence_log (track_id, ad_id, start_ts, end_ts, duration_sec, age)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(ev.get("track_id") or 0),
                        ad_id,
                        float(ev.get("start_ts") or 0.0),
                        float(ev.get("end_ts") or 0.0),
                        float(ev.get("duration_sec") or 0.0),
                        float(ev.get("age")) if ev.get("age") is not None else None,
                    ),
                )
        conn.commit()
        print("[API] ingest bulk commit ok")
        return {"status": "ok", "count": len(items)}
    finally:
        conn.close()