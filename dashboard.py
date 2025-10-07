import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from flask import Flask, jsonify, render_template_string, request


app = Flask(__name__)


def get_db():
    conn = sqlite3.connect("camera_analytics.db")
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def index():
    return render_template_string(
        DASHBOARD_HTML
    )


@app.route("/api/summary")
def api_summary():
    """Return aggregates for the selected time window."""
    window = request.args.get("window", "1h")
    now = datetime.now(timezone.utc)
    if window.endswith("h"):
        hours = int(window[:-1])
        start = now - timedelta(hours=hours)
    elif window.endswith("d"):
        days = int(window[:-1])
        start = now - timedelta(days=days)
    else:
        start = now - timedelta(hours=1)

    since_ts = start.timestamp()

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            COUNT(*) AS samples,
            COALESCE(SUM(new_unique_persons), 0) AS footfall_sum,
            COALESCE(SUM(new_unique_faces), 0) AS unique_faces_sum,
            MAX(unique_tracked_persons) AS unique_tracked_persons_max,
            COALESCE(SUM(new_gender_male), 0) AS male_sum,
            COALESCE(SUM(new_gender_female), 0) AS female_sum,
            COALESCE(SUM(new_gender_unknown), 0) AS unknown_sum,
            COALESCE(SUM(processing_time_ms), 0) AS sum_processing_ms
        FROM analytics
        WHERE ts >= ?
        """,
        (since_ts,),
    )
    row = cur.fetchone()
    conn.close()

    samples = int(row[0] or 0)
    sum_ms = float(row[7] or 0.0)
    avg_ms = (sum_ms / samples) if samples > 0 else 0.0
    est_fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0

    data: Dict[str, Any] = {
        "samples": samples,
        "footfall": int(row[1] or 0),
        "faces": int(row[2] or 0),
        "unique_tracked_persons": int(row[3] or 0),
        "gender": {
            "male": int(row[4] or 0),
            "female": int(row[5] or 0),
            "unknown": int(row[6] or 0),
        },
        "avg_processing_ms": avg_ms,
        "estimated_fps": est_fps,
    }
    return jsonify(data)


@app.route("/api/footfall_series")
def api_footfall_series():
    """Return time-series of footfall per minute for the selected window."""
    window = request.args.get("window", "1h")
    now = datetime.now(timezone.utc)
    if window.endswith("h"):
        hours = int(window[:-1])
        start = now - timedelta(hours=hours)
        bucket_seconds = 60
    elif window.endswith("d"):
        days = int(window[:-1])
        start = now - timedelta(days=days)
        bucket_seconds = 300  # 5 minutes
    else:
        start = now - timedelta(hours=1)
        bucket_seconds = 60

    since_ts = start.timestamp()

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT CAST(ts / ? AS INTEGER) * ? AS bucket, SUM(new_unique_persons) AS cnt
        FROM analytics
        WHERE ts >= ?
        GROUP BY bucket
        ORDER BY bucket
        """,
        (bucket_seconds, bucket_seconds, since_ts),
    )
    rows = cur.fetchall()
    conn.close()

    series = [
        {
            "t": int(b),
            "count": int(c or 0),
        }
        for (b, c) in rows
    ]
    return jsonify({"bucketSeconds": bucket_seconds, "series": series})


@app.route("/api/gender_series")
def api_gender_series():
    """Return time-series of gender counts (new only) for the selected window."""
    window = request.args.get("window", "1h")
    now = datetime.now(timezone.utc)
    if window.endswith("h"):
        hours = int(window[:-1])
        start = now - timedelta(hours=hours)
        bucket_seconds = 60
    elif window.endswith("d"):
        days = int(window[:-1])
        start = now - timedelta(days=days)
        bucket_seconds = 300
    else:
        start = now - timedelta(hours=1)
        bucket_seconds = 60

    since_ts = start.timestamp()

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT CAST(ts / ? AS INTEGER) * ? AS bucket,
               COALESCE(SUM(new_gender_male), 0) AS m,
               COALESCE(SUM(new_gender_female), 0) AS f,
               COALESCE(SUM(new_gender_unknown), 0) AS u
        FROM analytics
        WHERE ts >= ?
        GROUP BY bucket
        ORDER BY bucket
        """,
        (bucket_seconds, bucket_seconds, since_ts),
    )
    rows = cur.fetchall()
    conn.close()

    series = [
        {"t": int(b), "male": int(m or 0), "female": int(f or 0), "unknown": int(u or 0)}
        for (b, m, f, u) in rows
    ]
    return jsonify({"bucketSeconds": bucket_seconds, "series": series})


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Camera Analytics Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; background: #0f1116; color: #e6e6e6; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
    .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
    .card { background: #151823; border-radius: 10px; padding: 16px; }
    h1 { margin: 0 0 12px 0; font-size: 22px; }
    h2 { margin: 0 0 8px 0; font-size: 18px; }
    .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
    .kpi { background: #0c0f16; border-radius: 8px; padding: 12px; text-align: center; }
    .kpi .num { font-size: 26px; font-weight: 700; color: #4bd1ff; }
    .kpi .label { font-size: 12px; color: #9aa4b2; }
    .toolbar { margin-bottom: 12px; display: flex; justify-content: flex-end; }
    select { background: #0c0f16; color: #e6e6e6; border: 1px solid #2a2f3a; padding: 6px 10px; border-radius: 6px; }
    canvas { max-height: 360px; }
  </style>
  </head>
  <body>
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
      <h1>Camera Analytics</h1>
      <div class="toolbar">
        <label style="margin-right:8px;">Window:</label>
        <select id="window">
          <option value="1h">Last 1h</option>
          <option value="6h">Last 6h</option>
          <option value="24h">Last 24h</option>
          <option value="7d">Last 7d</option>
        </select>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>KPIs</h2>
        <div class="kpis">
          <div class="kpi"><div class="num" id="kpi-footfall">0</div><div class="label">Footfall (unique)</div></div>
          <div class="kpi"><div class="num" id="kpi-unique">0</div><div class="label">Unique Tracked</div></div>
          <div class="kpi"><div class="num" id="kpi-fps">0</div><div class="label">Estimated FPS</div></div>
        </div>
      </div>
    </div>

    <div class="row">
      <div class="card">
        <h2>Footfall Over Time</h2>
        <canvas id="footfallChart"></canvas>
      </div>
      <div class="card">
        <h2>Gender Distribution</h2>
        <canvas id="genderPieChart"></canvas>
      </div>
    </div>

    <script>
      const wnd = document.getElementById('window');
      let genderSeriesChart, genderPieChart, footfallChart;

      function fmtTs(t){
        const d = new Date(t * 1000);
        return d.toLocaleTimeString();
      }

      async function fetchJSON(url){
        const res = await fetch(url);
        return res.json();
      }

      function updateKPIs(sum){
        document.getElementById('kpi-footfall').textContent = sum.footfall;
        document.getElementById('kpi-unique').textContent = sum.unique_tracked_persons;
      }

      function renderGenderSeries(series){
        const labels = series.map(p => fmtTs(p.t));
        const dm = series.map(p => p.male);
        const df = series.map(p => p.female);
        const du = series.map(p => p.unknown);
        const data = {
          labels,
          datasets: [
            { label: 'Male', data: dm, borderColor: '#4bd1ff', backgroundColor: 'rgba(75,209,255,0.25)', fill: true, tension: 0.3 },
            { label: 'Female', data: df, borderColor: '#ff6fb1', backgroundColor: 'rgba(255,111,177,0.25)', fill: true, tension: 0.3 },
            { label: 'Unknown', data: du, borderColor: '#9aa4b2', backgroundColor: 'rgba(154,164,178,0.25)', fill: true, tension: 0.3 }
          ]
        };
        const cfg = { type: 'line', data, options: { plugins: { legend: { labels: { color: '#e6e6e6' } } }, scales: { x: { ticks: { color: '#9aa4b2' } }, y: { stacked: true, ticks: { color: '#9aa4b2' } } } } };
        if (genderSeriesChart) { genderSeriesChart.destroy(); }
        genderSeriesChart = new Chart(document.getElementById('genderSeriesChart'), cfg);
      }

      function renderGenderPie(g){
        const data = {
          labels: ['Male', 'Female', 'Unknown'],
          datasets: [{
            data: [g.male||0, g.female||0, g.unknown||0],
            backgroundColor: ['#4bd1ff', '#ff6fb1', '#9aa4b2'],
            borderColor: ['#2aa7d1', '#d85593', '#7f8895']
          }]
        };
        const cfg = { type: 'doughnut', data, options: { plugins: { legend: { labels: { color: '#e6e6e6' } } } } };
        if (genderPieChart) { genderPieChart.destroy(); }
        genderPieChart = new Chart(document.getElementById('genderPieChart'), cfg);
      }

      function renderFootfall(series, bucketSeconds){
        const labels = series.map(p => fmtTs(p.t));
        const counts = series.map(p => p.count);
        const data = {
          labels,
          datasets: [{
            label: 'Footfall',
            data: counts,
            fill: true,
            borderColor: '#4bd1ff',
            backgroundColor: 'rgba(75,209,255,0.15)',
            tension: 0.3,
          }]
        };
        const cfg = { type: 'line', data, options: { scales: { x: { ticks: { color: '#9aa4b2' } }, y: { ticks: { color: '#9aa4b2' } } }, plugins: { legend: { labels: { color: '#e6e6e6' } } } } };
        if (footfallChart) { footfallChart.destroy(); }
        footfallChart = new Chart(document.getElementById('footfallChart'), cfg);
      }

      async function refresh(){
        const w = wnd.value;
        const sum = await fetchJSON(`/api/summary?window=${w}`);
        updateKPIs(sum);
        const ts = await fetchJSON(`/api/footfall_series?window=${w}`);
        renderFootfall(ts.series || [], ts.bucketSeconds || 60);
        const gs = await fetchJSON(`/api/gender_series?window=${w}`);
        renderGenderSeries(gs.series || []);
        renderGenderPie(sum.gender || {});
      }

      wnd.addEventListener('change', refresh);
      setInterval(refresh, 5000);
      refresh();
    </script>
  </body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


