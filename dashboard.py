#!/usr/bin/env python3
"""PPE Monitoring Admin Dashboard — lightweight web UI for event monitoring.

Runs on Python's built-in http.server (no Flask/Django needed).
Reads the same events.csv that ppe_webcam_poc.py writes to.

Usage:
    python dashboard.py                        # default: events.csv, port 8080
    python dashboard.py --events events.csv --port 8080
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse, parse_qs

CSV_FIELDS = [
    "timestamp_utc", "camera", "track_id", "event_type", "crossing_direction",
    "helmet_status", "vest_status", "overall_status", "model", "evidence_image",
]


def read_events(csv_path: Path) -> List[Dict[str, str]]:
    """Read all events from the CSV file, newest first."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    rows: List[Dict[str, str]] = []
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    except Exception:
        return []
    rows.reverse()
    return rows


def compute_stats(events: List[Dict[str, str]]) -> Dict[str, Any]:
    """Compute summary statistics from events."""
    total = len(events)
    violations = sum(1 for e in events if e.get("event_type") == "PPE_NONCOMPLIANT_CROSSING")
    compliant = sum(1 for e in events if e.get("event_type") == "PPE_COMPLIANT_CROSSING")
    unknown = sum(1 for e in events if e.get("event_type") == "PPE_UNKNOWN_AT_CROSSING")

    cameras: Dict[str, Dict[str, int]] = {}
    for e in events:
        cam = e.get("camera", "unknown")
        if cam not in cameras:
            cameras[cam] = {"total": 0, "violations": 0, "compliant": 0, "unknown": 0}
        cameras[cam]["total"] += 1
        if e.get("event_type") == "PPE_NONCOMPLIANT_CROSSING":
            cameras[cam]["violations"] += 1
        elif e.get("event_type") == "PPE_COMPLIANT_CROSSING":
            cameras[cam]["compliant"] += 1
        elif e.get("event_type") == "PPE_UNKNOWN_AT_CROSSING":
            cameras[cam]["unknown"] += 1

    compliance_rate = (compliant / total * 100) if total > 0 else 0.0

    return {
        "total_crossings": total,
        "violations": violations,
        "compliant": compliant,
        "unknown": unknown,
        "compliance_rate": round(compliance_rate, 1),
        "cameras": cameras,
        "server_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PPE Monitoring — Admin Dashboard</title>
<meta name="description" content="Real-time PPE compliance monitoring dashboard for factory safety">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.7);
    --border-color: rgba(255,255,255,0.06);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #3b82f6;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-purple: #8b5cf6;
    --glow-green: rgba(16,185,129,0.15);
    --glow-red: rgba(239,68,68,0.15);
    --glow-amber: rgba(245,158,11,0.15);
    --glow-blue: rgba(59,130,246,0.15);
  }

  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    overflow-x: hidden;
  }

  body::before {
    content: '';
    position: fixed;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 20% 50%, rgba(59,130,246,0.04) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(139,92,246,0.04) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(16,185,129,0.03) 0%, transparent 50%);
    z-index: 0;
    pointer-events: none;
  }

  .container { max-width: 1680px; width: 96%; margin: 0 auto; padding: 24px 28px; position: relative; z-index: 1; }

  /* Header */
  .header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 32px; flex-wrap: wrap; gap: 12px;
  }
  .header-left { display: flex; align-items: center; gap: 14px; }
  .logo-icon {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; box-shadow: 0 4px 20px rgba(59,130,246,0.25);
  }
  .header h1 { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.5px; }
  .header h1 span { color: var(--accent-blue); }
  .header-subtitle { font-size: 0.78rem; color: var(--text-muted); font-weight: 500; }
  .live-badge {
    display: flex; align-items: center; gap: 7px;
    background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.25);
    padding: 6px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;
    color: var(--accent-green);
  }
  .live-dot {
    width: 8px; height: 8px; border-radius: 50%; background: var(--accent-green);
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.5); } 50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(16,185,129,0); } }

  /* POC Banner */
  .poc-banner {
    background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2);
    border-radius: 10px; padding: 10px 18px; margin-bottom: 24px;
    font-size: 0.8rem; color: var(--accent-amber); font-weight: 500;
    display: flex; align-items: center; gap: 8px;
  }

  /* Stats Grid */
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 18px; margin-bottom: 28px; }
  .stat-card {
    background: var(--bg-card); backdrop-filter: blur(12px);
    border: 1px solid var(--border-color); border-radius: 16px;
    padding: 22px 24px; position: relative; overflow: hidden;
    transition: transform 0.2s, border-color 0.3s;
  }
  .stat-card:hover { transform: translateY(-2px); border-color: rgba(255,255,255,0.12); }
  .stat-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 16px 16px 0 0;
  }
  .stat-card.green::before { background: linear-gradient(90deg, var(--accent-green), #34d399); }
  .stat-card.red::before { background: linear-gradient(90deg, var(--accent-red), #f87171); }
  .stat-card.amber::before { background: linear-gradient(90deg, var(--accent-amber), #fbbf24); }
  .stat-card.blue::before { background: linear-gradient(90deg, var(--accent-blue), #60a5fa); }
  .stat-card.green { box-shadow: inset 0 1px 30px var(--glow-green); }
  .stat-card.red { box-shadow: inset 0 1px 30px var(--glow-red); }
  .stat-card.amber { box-shadow: inset 0 1px 30px var(--glow-amber); }
  .stat-card.blue { box-shadow: inset 0 1px 30px var(--glow-blue); }
  .stat-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 8px; }
  .stat-value { font-size: 2.2rem; font-weight: 800; letter-spacing: -1px; line-height: 1; }
  .stat-value.green { color: var(--accent-green); }
  .stat-value.red { color: var(--accent-red); }
  .stat-value.amber { color: var(--accent-amber); }
  .stat-value.blue { color: var(--accent-blue); }
  .stat-sub { font-size: 0.72rem; color: var(--text-muted); margin-top: 6px; font-weight: 500; }

  /* Two-column layout */
  .main-grid { display: grid; grid-template-columns: 1fr 360px; gap: 22px; }
  @media (max-width: 960px) { .main-grid { grid-template-columns: 1fr; } }

  /* Section card */
  .section {
    background: var(--bg-card); backdrop-filter: blur(12px);
    border: 1px solid var(--border-color); border-radius: 16px;
    overflow: hidden;
  }
  .section-header {
    padding: 16px 22px; border-bottom: 1px solid var(--border-color);
    display: flex; align-items: center; justify-content: space-between;
  }
  .section-title { font-size: 0.88rem; font-weight: 700; }
  .section-badge {
    font-size: 0.68rem; padding: 3px 10px; border-radius: 10px;
    font-weight: 600;
  }
  .section-badge.count { background: rgba(59,130,246,0.15); color: var(--accent-blue); }

  /* Event table */
  .events-table { width: 100%; border-collapse: collapse; }
  .events-table th {
    text-align: left; padding: 10px 16px; font-size: 0.7rem;
    color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.8px;
    font-weight: 600; border-bottom: 1px solid var(--border-color);
    position: sticky; top: 0; background: var(--bg-secondary);
  }
  .events-table td {
    padding: 10px 16px; font-size: 0.8rem; border-bottom: 1px solid var(--border-color);
    color: var(--text-secondary); white-space: nowrap;
  }
  .events-table tr { transition: background 0.15s; }
  .events-table tr:hover { background: rgba(255,255,255,0.02); }
  .events-table tr.violation { background: rgba(239,68,68,0.04); }
  .events-table tr.violation:hover { background: rgba(239,68,68,0.08); }
  .events-scroll { max-height: 420px; overflow-y: auto; }
  .events-scroll::-webkit-scrollbar { width: 5px; }
  .events-scroll::-webkit-scrollbar-track { background: transparent; }
  .events-scroll::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }

  /* Tags */
  .tag {
    display: inline-block; padding: 2px 9px; border-radius: 6px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.3px;
  }
  .tag-red { background: rgba(239,68,68,0.15); color: #f87171; }
  .tag-green { background: rgba(16,185,129,0.15); color: #34d399; }
  .tag-amber { background: rgba(245,158,11,0.15); color: #fbbf24; }
  .tag-gray { background: rgba(100,116,139,0.15); color: #94a3b8; }

  /* Camera cards */
  .camera-list { padding: 14px; display: flex; flex-direction: column; gap: 12px; }
  .camera-card {
    background: rgba(255,255,255,0.02); border: 1px solid var(--border-color);
    border-radius: 12px; padding: 16px; transition: border-color 0.2s;
  }
  .camera-card:hover { border-color: rgba(255,255,255,0.1); }
  .camera-name { font-size: 0.82rem; font-weight: 700; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
  .camera-name .cam-icon { font-size: 1rem; }
  .camera-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .cam-stat { text-align: center; padding: 8px; background: rgba(0,0,0,0.2); border-radius: 8px; }
  .cam-stat-val { font-size: 1.2rem; font-weight: 800; }
  .cam-stat-label { font-size: 0.62rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }

  /* Empty state */
  .empty-state {
    padding: 60px 20px; text-align: center; color: var(--text-muted);
  }
  .empty-icon { font-size: 3rem; margin-bottom: 12px; opacity: 0.4; }
  .empty-text { font-size: 0.88rem; font-weight: 500; }
  .empty-sub { font-size: 0.75rem; margin-top: 6px; }

  /* Footer */
  .footer {
    text-align: center; padding: 24px; font-size: 0.7rem;
    color: var(--text-muted); margin-top: 20px;
  }

  /* Alert Banner */
  .alert-banner {
    background: linear-gradient(90deg, rgba(239,68,68,0.25), rgba(185,28,28,0.35));
    border: 1px solid rgba(239,68,68,0.6);
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 0 25px rgba(239,68,68,0.25);
    animation: alertPulse 1.5s infinite;
  }
  @keyframes alertPulse {
    0%, 100% { box-shadow: 0 0 15px rgba(239,68,68,0.25); border-color: rgba(239,68,68,0.5); }
    50% { box-shadow: 0 0 35px rgba(239,68,68,0.6); border-color: rgba(239,68,68,0.9); }
  }
  .alert-icon { font-size: 2rem; }
  .alert-title { font-weight: 800; font-size: 0.95rem; color: #f87171; letter-spacing: 0.3px; }
  .alert-desc { font-size: 0.8rem; color: #fca5a5; margin-top: 2px; }
  .alert-meta { margin-left: auto; font-size: 0.75rem; color: #f87171; font-weight: 600; text-align: right; }

  /* Refresh animation */
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
  .fade-in { animation: fadeIn 0.3s ease-out; }
</style>
</head>
<body>
<div class="container">
  <header class="header">
    <div class="header-left">
      <div class="logo-icon">🛡️</div>
      <div>
        <h1>PPE <span>Monitor</span></h1>
        <div class="header-subtitle">Admin Dashboard · Real-time Event Feed</div>
      </div>
    </div>
    <div class="live-badge"><div class="live-dot"></div> LIVE — Auto-refresh 2s</div>
  </header>

  <div class="alert-banner" id="alert-banner" style="display: none;">
    <div class="alert-icon">🚨</div>
    <div>
      <div class="alert-title" id="alert-title">NON-COMPLIANT WORKER CROSSING DETECTED</div>
      <div class="alert-desc" id="alert-desc">An unauthorized entry was logged without required PPE equipment.</div>
    </div>
    <div class="alert-meta">
      <div id="alert-cam">CAM 0</div>
      <div id="alert-time" style="font-weight: 400; opacity: 0.85;">Just now</div>
    </div>
  </div>

  <div class="poc-banner">
    ⚠️ POC ONLY — This dashboard is a monitoring aid, not a safety-rated system. Do not use as the sole safety barrier.
  </div>

  <div class="stats-grid" id="stats-grid">
    <div class="stat-card blue"><div class="stat-label">Total Crossings</div><div class="stat-value blue" id="stat-total">—</div><div class="stat-sub">All detected line crossings</div></div>
    <div class="stat-card red"><div class="stat-label">Violations</div><div class="stat-value red" id="stat-violations">—</div><div class="stat-sub">Non-compliant crossings</div></div>
    <div class="stat-card green"><div class="stat-label">Compliance Rate</div><div class="stat-value green" id="stat-rate">—</div><div class="stat-sub">Of resolved crossings</div></div>
    <div class="stat-card amber"><div class="stat-label">Unknown</div><div class="stat-value amber" id="stat-unknown">—</div><div class="stat-sub">Insufficient PPE evidence</div></div>
  </div>

  <div class="main-grid">
    <div class="section">
      <div class="section-header">
        <div class="section-title">📋 Recent Events</div>
        <div class="section-badge count" id="event-count">0 events</div>
      </div>
      <div class="events-scroll" id="events-container">
        <div class="empty-state" id="empty-state">
          <div class="empty-icon">📷</div>
          <div class="empty-text">No crossing events yet</div>
          <div class="empty-sub">Events will appear here when workers cross the entry line</div>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-header">
        <div class="section-title">📹 Cameras</div>
      </div>
      <div class="camera-list" id="camera-list">
        <div class="empty-state">
          <div class="empty-icon">📡</div>
          <div class="empty-text">No cameras reporting</div>
          <div class="empty-sub">Start ppe_webcam_poc.py to begin</div>
        </div>
      </div>
    </div>
  </div>

  <footer class="footer">
    PPE Webcam Monitoring POC · Dashboard v1.0 · Last update: <span id="last-update">—</span>
  </footer>
</div>

<script>
const API_URL = '/api/events';

function eventTypeTag(type) {
  if (type === 'PPE_NONCOMPLIANT_CROSSING') return '<span class="tag tag-red">NON-COMPLIANT</span>';
  if (type === 'PPE_COMPLIANT_CROSSING') return '<span class="tag tag-green">COMPLIANT</span>';
  if (type === 'PPE_UNKNOWN_AT_CROSSING') return '<span class="tag tag-amber">UNKNOWN</span>';
  return `<span class="tag tag-gray">${type}</span>`;
}

function statusTag(status) {
  if (status === 'PRESENT') return '<span class="tag tag-green">✓</span>';
  if (status === 'MISSING') return '<span class="tag tag-red">✗</span>';
  return '<span class="tag tag-gray">?</span>';
}

function formatTime(ts) {
  if (!ts) return '—';
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});
  } catch { return ts; }
}

function renderEvents(events) {
  const container = document.getElementById('events-container');
  const emptyState = document.getElementById('empty-state');

  if (!events || events.length === 0) {
    container.innerHTML = '';
    container.appendChild(emptyState);
    emptyState.style.display = 'block';
    return;
  }

  let html = '<table class="events-table"><thead><tr>';
  html += '<th>Time</th><th>Camera</th><th>Track</th><th>Status</th><th>Helmet</th><th>Vest</th><th>Direction</th>';
  html += '</tr></thead><tbody>';

  const show = events.slice(0, 50);
  for (const e of show) {
    const rowClass = e.event_type === 'PPE_NONCOMPLIANT_CROSSING' ? ' class="violation"' : '';
    html += `<tr${rowClass}>`;
    html += `<td>${formatTime(e.timestamp_utc)}</td>`;
    html += `<td>Cam ${e.camera || '?'}</td>`;
    html += `<td>#${e.track_id || '?'}</td>`;
    html += `<td>${eventTypeTag(e.event_type)}</td>`;
    html += `<td>${statusTag(e.helmet_status)}</td>`;
    html += `<td>${statusTag(e.vest_status)}</td>`;
    html += `<td>${e.crossing_direction || '—'}</td>`;
    html += '</tr>';
  }
  html += '</tbody></table>';
  container.innerHTML = html;
}

function renderCameras(cameras) {
  const container = document.getElementById('camera-list');
  if (!cameras || Object.keys(cameras).length === 0) {
    container.innerHTML = '<div class="empty-state"><div class="empty-icon">📡</div><div class="empty-text">No cameras reporting</div><div class="empty-sub">Start ppe_webcam_poc.py to begin</div></div>';
    return;
  }

  let html = '';
  for (const [name, stats] of Object.entries(cameras)) {
    html += `<div class="camera-card fade-in">`;
    html += `<div class="camera-name"><span class="cam-icon">📷</span> Camera ${name}</div>`;
    html += '<div class="camera-stats">';
    html += `<div class="cam-stat"><div class="cam-stat-val blue">${stats.total}</div><div class="cam-stat-label">Total</div></div>`;
    html += `<div class="cam-stat"><div class="cam-stat-val red">${stats.violations}</div><div class="cam-stat-label">Violations</div></div>`;
    html += `<div class="cam-stat"><div class="cam-stat-val green">${stats.compliant}</div><div class="cam-stat-label">Compliant</div></div>`;
    html += `<div class="cam-stat"><div class="cam-stat-val amber">${stats.unknown}</div><div class="cam-stat-label">Unknown</div></div>`;
    html += '</div></div>';
  }
  container.innerHTML = html;
}

let previousViolations = null;

function playAlertChime() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.35);
    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.36);
  } catch (err) {}
}

async function refresh() {
  try {
    const resp = await fetch(API_URL);
    if (!resp.ok) return;
    const data = await resp.json();

    document.getElementById('stat-total').textContent = data.stats.total_crossings;
    document.getElementById('stat-violations').textContent = data.stats.violations;
    document.getElementById('stat-rate').textContent = data.stats.compliance_rate + '%';
    document.getElementById('stat-unknown').textContent = data.stats.unknown;
    document.getElementById('event-count').textContent = data.stats.total_crossings + ' events';
    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();

    // Check for recent violations
    const banner = document.getElementById('alert-banner');
    const recentViolations = (data.events || []).filter(e => e.event_type === 'PPE_NONCOMPLIANT_CROSSING');
    if (recentViolations.length > 0) {
      const latest = recentViolations[0];
      const missingItems = [];
      if (latest.helmet_status === 'MISSING') missingItems.push('Hardhat');
      if (latest.vest_status === 'MISSING') missingItems.push('Safety Vest');
      const missingStr = missingItems.length > 0 ? missingItems.join(' and ') : 'required PPE items';

      document.getElementById('alert-title').textContent = `🚨 REAL-TIME ALERT: WORKER #${latest.track_id || '?'} VIOLATION`;
      document.getElementById('alert-desc').textContent = `Non-compliant entry recorded on Camera ${latest.camera || '0'} — Missing ${missingStr}.`;
      document.getElementById('alert-cam').textContent = `CAM ${latest.camera || '0'}`;
      document.getElementById('alert-time').textContent = formatTime(latest.timestamp_utc);
      banner.style.display = 'flex';

      // Play alert chime if a new violation arrived
      if (previousViolations !== null && data.stats.violations > previousViolations) {
        playAlertChime();
      }
    } else {
      banner.style.display = 'none';
    }
    previousViolations = data.stats.violations;

    renderEvents(data.events);
    renderCameras(data.stats.cameras);
  } catch (err) {
    console.error('Dashboard refresh error:', err);
  }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves the dashboard HTML and a JSON API for events."""

    csv_path: Path = Path("events.csv")

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy request logs; keep errors.
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/dashboard":
            self._serve_html()
        elif parsed.path == "/api/events":
            self._serve_api()
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        content = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def _serve_api(self) -> None:
        events = read_events(self.csv_path)
        stats = compute_stats(events)
        payload = json.dumps({"events": events, "stats": stats}, ensure_ascii=False)
        content = payload.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="PPE Monitoring Admin Dashboard")
    parser.add_argument("--events", default="events.csv", help="Path to the events CSV file")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port")
    args = parser.parse_args()

    DashboardHandler.csv_path = Path(args.events)
    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"PPE Admin Dashboard running at http://localhost:{args.port}")
    print(f"Reading events from: {Path(args.events).resolve()}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
