#!/usr/bin/env python3
"""PPE Monitoring Admin Dashboard — Enterprise web UI for event and camera monitoring.

Features:
- Dynamic IP CCTV (RTSP/HTTP) and USB Camera Discovery & Status Monitoring
- Live connection state indicators (CONNECTED / NOT CONNECTED / RECONNECTING)
- Automatic status polling without page reloads
- Real-time PPE compliance violation alerts with audio chime
- Dynamic hardware scan trigger via /api/cameras/scan
- Filter cameras: Show All, Show Connected Only, Show Disconnected

Usage:
    python dashboard.py                        # default: events.csv, port 8080
    python dashboard.py --events events.csv --port 8080
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

from camera_manager import CameraManager

logger = logging.getLogger("Dashboard")

CSV_FIELDS = [
    "timestamp_utc", "camera", "track_id", "event_type", "crossing_direction",
    "helmet_status", "vest_status", "overall_status", "model", "evidence_image",
]

camera_manager = CameraManager()


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
<title>PPE Monitoring & CCTV Command Center</title>
<meta name="description" content="Real-time PPE compliance monitoring & CCTV camera health dashboard">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg-primary: #070a13;
    --bg-secondary: #0f172a;
    --bg-card: rgba(15, 23, 42, 0.75);
    --border-color: rgba(255,255,255,0.08);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #3b82f6;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-purple: #8b5cf6;
    --glow-green: rgba(16,185,129,0.18);
    --glow-red: rgba(239,68,68,0.18);
    --glow-amber: rgba(245,158,11,0.18);
    --glow-blue: rgba(59,130,246,0.18);
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
    background: radial-gradient(ellipse at 20% 30%, rgba(59,130,246,0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(139,92,246,0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(16,185,129,0.04) 0%, transparent 50%);
    z-index: 0;
    pointer-events: none;
  }

  .container { max-width: 1720px; width: 96%; margin: 0 auto; padding: 24px 20px; position: relative; z-index: 1; }

  /* Header */
  .header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 24px; flex-wrap: wrap; gap: 14px;
  }
  .header-left { display: flex; align-items: center; gap: 14px; }
  .logo-icon {
    width: 48px; height: 48px; border-radius: 12px;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; box-shadow: 0 4px 20px rgba(59,130,246,0.3);
  }
  .header h1 { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.5px; }
  .header h1 span { color: var(--accent-blue); }
  .header-subtitle { font-size: 0.8rem; color: var(--text-muted); font-weight: 500; }
  
  .header-actions { display: flex; align-items: center; gap: 12px; }
  .live-badge {
    display: flex; align-items: center; gap: 8px;
    background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3);
    padding: 7px 16px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    color: var(--accent-green);
  }
  .live-dot {
    width: 9px; height: 9px; border-radius: 50%; background: var(--accent-green);
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.6); } 50% { opacity: 0.7; box-shadow: 0 0 0 7px rgba(16,185,129,0); } }

  .btn-scan {
    display: flex; align-items: center; gap: 8px;
    background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.35);
    color: #60a5fa; padding: 7px 16px; border-radius: 20px; font-size: 0.8rem;
    font-weight: 600; cursor: pointer; transition: all 0.2s;
  }
  .btn-scan:hover { background: rgba(59,130,246,0.25); border-color: #60a5fa; transform: translateY(-1px); }
  .btn-scan.scanning { opacity: 0.7; pointer-events: none; }
  .btn-scan.scanning .scan-icon { animation: spin 1s linear infinite; }
  @keyframes spin { 100% { transform: rotate(360deg); } }

  /* Alert Banner */
  .alert-banner {
    background: linear-gradient(90deg, rgba(239,68,68,0.25), rgba(185,28,28,0.35));
    border: 1px solid rgba(239,68,68,0.6);
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 22px;
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
  .alert-desc { font-size: 0.82rem; color: #fca5a5; margin-top: 2px; }
  .alert-meta { margin-left: auto; font-size: 0.75rem; color: #f87171; font-weight: 600; text-align: right; }

  /* Stats Grid */
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .stat-card {
    background: var(--bg-card); backdrop-filter: blur(12px);
    border: 1px solid var(--border-color); border-radius: 14px;
    padding: 20px 22px; position: relative; overflow: hidden;
    transition: transform 0.2s, border-color 0.3s;
  }
  .stat-card:hover { transform: translateY(-2px); border-color: rgba(255,255,255,0.14); }
  .stat-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 14px 14px 0 0;
  }
  .stat-card.green::before { background: linear-gradient(90deg, var(--accent-green), #34d399); }
  .stat-card.red::before { background: linear-gradient(90deg, var(--accent-red), #f87171); }
  .stat-card.amber::before { background: linear-gradient(90deg, var(--accent-amber), #fbbf24); }
  .stat-card.blue::before { background: linear-gradient(90deg, var(--accent-blue), #60a5fa); }
  .stat-card.green { box-shadow: inset 0 1px 30px var(--glow-green); }
  .stat-card.red { box-shadow: inset 0 1px 30px var(--glow-red); }
  .stat-card.amber { box-shadow: inset 0 1px 30px var(--glow-amber); }
  .stat-card.blue { box-shadow: inset 0 1px 30px var(--glow-blue); }
  .stat-label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 8px; }
  .stat-value { font-size: 2.1rem; font-weight: 800; letter-spacing: -1px; line-height: 1; }
  .stat-value.green { color: var(--accent-green); }
  .stat-value.red { color: var(--accent-red); }
  .stat-value.amber { color: var(--accent-amber); }
  .stat-value.blue { color: var(--accent-blue); }
  .stat-sub { font-size: 0.72rem; color: var(--text-muted); margin-top: 6px; font-weight: 500; }

  /* Camera Health Section */
  .camera-section {
    background: var(--bg-card); backdrop-filter: blur(12px);
    border: 1px solid var(--border-color); border-radius: 16px;
    margin-bottom: 24px; overflow: hidden;
  }
  .camera-header {
    padding: 16px 22px; border-bottom: 1px solid var(--border-color);
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;
  }
  .camera-title-wrap { display: flex; align-items: center; gap: 10px; }
  .camera-title { font-size: 0.95rem; font-weight: 700; }
  .camera-filter-pills { display: flex; gap: 8px; }
  .filter-pill {
    padding: 4px 12px; border-radius: 14px; font-size: 0.72rem; font-weight: 600;
    background: rgba(255,255,255,0.04); border: 1px solid var(--border-color);
    color: var(--text-secondary); cursor: pointer; transition: all 0.15s;
  }
  .filter-pill:hover, .filter-pill.active {
    background: rgba(59,130,246,0.18); border-color: rgba(59,130,246,0.4); color: #93c5fd;
  }

  .camera-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 16px; padding: 20px;
  }
  .camera-card {
    background: rgba(255,255,255,0.02); border: 1px solid var(--border-color);
    border-radius: 12px; padding: 18px; position: relative; overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
  }
  .camera-card:hover { transform: translateY(-2px); border-color: rgba(255,255,255,0.14); }
  .camera-card.connected { border-left: 4px solid var(--accent-green); }
  .camera-card.not-connected { border-left: 4px solid var(--accent-red); background: rgba(239,68,68,0.02); }
  .camera-card.reconnecting { border-left: 4px solid var(--accent-amber); }

  .cam-top-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
  .cam-title { font-size: 0.9rem; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 8px; }
  .cam-type-badge {
    font-size: 0.65rem; padding: 2px 7px; border-radius: 6px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.5px;
    background: rgba(255,255,255,0.06); color: var(--text-secondary);
  }

  .cam-status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.3px;
  }
  .cam-status-pill.connected { background: rgba(16,185,129,0.15); color: #34d399; }
  .cam-status-pill.not-connected { background: rgba(239,68,68,0.15); color: #f87171; }
  .cam-status-pill.reconnecting { background: rgba(245,158,11,0.15); color: #fbbf24; }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; }
  .status-dot.green { background: #34d399; box-shadow: 0 0 8px #34d399; }
  .status-dot.red { background: #f87171; box-shadow: 0 0 8px #f87171; }
  .status-dot.amber { background: #fbbf24; animation: blink 1s infinite; }
  @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

  .cam-meta-row { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 12px; font-family: monospace; word-break: break-all; }
  .cam-metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px; }
  .cam-metric { background: rgba(0,0,0,0.25); padding: 8px; border-radius: 8px; text-align: center; }
  .cam-metric-val { font-size: 0.85rem; font-weight: 700; color: var(--text-primary); }
  .cam-metric-lbl { font-size: 0.62rem; color: var(--text-muted); text-transform: uppercase; margin-top: 2px; }

  .cam-error-box {
    margin-top: 10px; padding: 8px 10px; border-radius: 8px;
    background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2);
    font-size: 0.72rem; color: #fca5a5; display: flex; align-items: center; gap: 6px;
  }

  /* Two-column layout */
  .main-grid { display: grid; grid-template-columns: 1fr 380px; gap: 20px; }
  @media (max-width: 1080px) { .main-grid { grid-template-columns: 1fr; } }

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
  .section-title { font-size: 0.9rem; font-weight: 700; }
  .section-badge {
    font-size: 0.7rem; padding: 3px 10px; border-radius: 10px;
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
  .events-scroll { max-height: 480px; overflow-y: auto; }
  .events-scroll::-webkit-scrollbar { width: 5px; }
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

  /* Empty state */
  .empty-state {
    padding: 50px 20px; text-align: center; color: var(--text-muted);
  }
  .empty-icon { font-size: 2.8rem; margin-bottom: 10px; opacity: 0.4; }
  .empty-text { font-size: 0.88rem; font-weight: 500; }
  .empty-sub { font-size: 0.75rem; margin-top: 4px; }

  /* Footer */
  .footer {
    text-align: center; padding: 24px; font-size: 0.72rem;
    color: var(--text-muted); margin-top: 24px;
  }
</style>
</head>
<body>
<div class="container">
  <header class="header">
    <div class="header-left">
      <div class="logo-icon">🛡️</div>
      <div>
        <h1>PPE & CCTV <span>Command Center</span></h1>
        <div class="header-subtitle">Enterprise Computer Vision · YOLOv8 & OpenCV Surveillance Engine</div>
      </div>
    </div>
    <div class="header-actions">
      <button class="btn-scan" id="btn-scan" onclick="triggerDynamicScan()">
        <span class="scan-icon">🔄</span> Scan Hardware & Ports
      </button>
      <div class="live-badge"><div class="live-dot"></div> LIVE MONITORING</div>
    </div>
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

  <!-- Camera Hardware Section -->
  <div class="camera-section">
    <div class="camera-header">
      <div class="camera-title-wrap">
        <span style="font-size: 1.2rem;">📹</span>
        <div class="camera-title">Dynamic Camera Network & CCTV Feeds</div>
        <span class="section-badge count" id="cam-summary-badge">0 / 0 Connected</span>
      </div>
      <div class="camera-filter-pills">
        <button class="filter-pill active" onclick="setCameraFilter('connected')">Connected Only (<span id="count-connected">0</span>)</button>
        <button class="filter-pill" onclick="setCameraFilter('all')">All Configured (<span id="count-all">0</span>)</button>
        <button class="filter-pill" onclick="setCameraFilter('disconnected')">Not Connected (<span id="count-disconnected">0</span>)</button>
      </div>
    </div>
    <div class="camera-grid" id="camera-grid">
      <div class="empty-state">
        <div class="empty-icon">📡</div>
        <div class="empty-text">Scanning for cameras...</div>
      </div>
    </div>
  </div>

  <!-- Compliance Stats Grid -->
  <div class="stats-grid" id="stats-grid">
    <div class="stat-card blue"><div class="stat-label">Total Crossings</div><div class="stat-value blue" id="stat-total">—</div><div class="stat-sub">Tracked entry crossings</div></div>
    <div class="stat-card red"><div class="stat-label">Violations</div><div class="stat-value red" id="stat-violations">—</div><div class="stat-sub">Non-compliant events</div></div>
    <div class="stat-card green"><div class="stat-label">Compliance Rate</div><div class="stat-value green" id="stat-rate">—</div><div class="stat-sub">PPE adherence percentage</div></div>
    <div class="stat-card amber"><div class="stat-label">Unknown Evidence</div><div class="stat-value amber" id="stat-unknown">—</div><div class="stat-sub">Unresolved PPE state</div></div>
  </div>

  <div class="main-grid">
    <!-- Events Feed -->
    <div class="section">
      <div class="section-header">
        <div class="section-title">📋 Live Safety Event Stream</div>
        <div class="section-badge count" id="event-count">0 events</div>
      </div>
      <div class="events-scroll" id="events-container">
        <div class="empty-state" id="empty-state">
          <div class="empty-icon">📷</div>
          <div class="empty-text">No crossing events logged yet</div>
          <div class="empty-sub">Events appear automatically as workers enter monitored zones</div>
        </div>
      </div>
    </div>

    <!-- Active Camera Metrics & Zone Settings -->
    <div class="section">
      <div class="section-header">
        <div class="section-title">⚙️ AI Inference & Zone Engine</div>
      </div>
      <div style="padding: 20px;">
        <div style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.6;">
          <div style="margin-bottom: 12px;"><strong>AI Model:</strong> <span style="color: var(--accent-blue);">Self-Trained YOLOv8 (weights/my_custom_ppe.pt)</span></div>
          <div style="margin-bottom: 12px;"><strong>Vision Backend:</strong> <span style="color: #34d399;">OpenCV 5.0 + CUDA Acceleration</span></div>
          <div style="margin-bottom: 12px;"><strong>Tracking Algorithm:</strong> ByteTrack Multi-Worker Association</div>
          <div style="margin-bottom: 12px;"><strong>Zone Security:</strong> Dual-Vertical Boundary (Left & Right Entry)</div>
          <div style="margin-bottom: 12px;"><strong>Auto-Reconnect:</strong> Exponential Backoff (3s - 10s)</div>
          <div style="margin-bottom: 12px;"><strong>Far Distance Zoom:</strong> OpenCV Intelligent Crop Inference Active</div>
        </div>
      </div>
    </div>
  </div>

  <footer class="footer">
    PPE Safety Surveillance Command Center · Dynamic RTSP & USB Camera Architecture · Last update: <span id="last-update">—</span>
  </footer>
</div>

<script>
let currentFilter = 'connected';
let cameraData = [];
let previousViolations = null;

function setCameraFilter(filter) {
  currentFilter = filter;
  document.querySelectorAll('.filter-pill').forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');
  renderCameraCards();
}

async function triggerDynamicScan() {
  const btn = document.getElementById('btn-scan');
  btn.classList.add('scanning');
  try {
    const resp = await fetch('/api/cameras/scan');
    if (resp.ok) {
      await refresh();
    }
  } catch (err) {
    console.error('Scan error:', err);
  } finally {
    btn.classList.remove('scanning');
  }
}

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
    html += `<td>${e.camera || '?'}</td>`;
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

function renderCameraCards() {
  const grid = document.getElementById('camera-grid');
  let list = cameraData || [];

  const total = list.length;
  const connected = list.filter(c => c.status === 'CONNECTED').length;
  const disconnected = total - connected;

  document.getElementById('count-all').textContent = total;
  document.getElementById('count-connected').textContent = connected;
  document.getElementById('count-disconnected').textContent = disconnected;
  document.getElementById('cam-summary-badge').textContent = `${connected} / ${total} Connected`;

  if (currentFilter === 'connected') {
    list = list.filter(c => c.status === 'CONNECTED');
  } else if (currentFilter === 'disconnected') {
    list = list.filter(c => c.status !== 'CONNECTED');
  }

  if (list.length === 0) {
    grid.innerHTML = '<div class="empty-state" style="grid-column: 1/-1;"><div class="empty-icon">🔍</div><div class="empty-text">No cameras matching current filter</div></div>';
    return;
  }

  let html = '';
  for (const cam of list) {
    const isConn = cam.status === 'CONNECTED';
    const isReconn = cam.status === 'RECONNECTING';
    const cardClass = isConn ? 'connected' : (isReconn ? 'reconnecting' : 'not-connected');
    const dotClass = isConn ? 'green' : (isReconn ? 'amber' : 'red');
    const statusText = isConn ? 'CONNECTED' : (isReconn ? 'RECONNECTING' : 'NOT CONNECTED');

    let maskedSource = cam.source || '';
    if (maskedSource.includes('@')) {
      maskedSource = maskedSource.replace(/:([^@:]+)@/, ':****@');
    }

    const res = (cam.resolution && cam.resolution[0]) ? `${cam.resolution[0]}x${cam.resolution[1]}` : '—';
    const fps = cam.fps > 0 ? `${cam.fps} FPS` : (isConn ? 'Active' : '0 FPS');

    html += `<div class="camera-card ${cardClass}">`;
    html += `  <div class="cam-top-row">`;
    html += `    <div class="cam-title"><span>${cam.type === 'ip' ? '🌐' : (cam.type === 'usb' ? '📷' : '🎞️')}</span> ${cam.name}</div>`;
    html += `    <span class="cam-type-badge">${cam.type}</span>`;
    html += `  </div>`;
    html += `  <div class="cam-top-row">`;
    html += `    <div class="cam-status-pill ${cardClass}"><div class="status-dot ${dotClass}"></div> ${statusText}</div>`;
    html += `    <span style="font-size: 0.7rem; color: var(--text-muted);">${cam.location || 'General'}</span>`;
    html += `  </div>`;
    html += `  <div class="cam-meta-row">${maskedSource}</div>`;

    if (!isConn && cam.error_message) {
      html += `  <div class="cam-error-box"><span>⚠️</span> ${cam.error_message}</div>`;
    }

    html += `  <div class="cam-metrics-grid">`;
    html += `    <div class="cam-metric"><div class="cam-metric-val">${res}</div><div class="cam-metric-lbl">Resolution</div></div>`;
    html += `    <div class="cam-metric"><div class="cam-metric-val">${fps}</div><div class="cam-metric-lbl">Frame Rate</div></div>`;
    html += `    <div class="cam-metric"><div class="cam-metric-val">${cam.last_seen_str || 'Never'}</div><div class="cam-metric-lbl">Last Seen</div></div>`;
    html += `  </div>`;
    html += `</div>`;
  }

  grid.innerHTML = html;
}

async function refresh() {
  try {
    // 1. Fetch Events & Stats
    const respEvents = await fetch('/api/events');
    if (respEvents.ok) {
      const data = await respEvents.json();
      document.getElementById('stat-total').textContent = data.stats.total_crossings;
      document.getElementById('stat-violations').textContent = data.stats.violations;
      document.getElementById('stat-rate').textContent = data.stats.compliance_rate + '%';
      document.getElementById('stat-unknown').textContent = data.stats.unknown;
      document.getElementById('event-count').textContent = data.stats.total_crossings + ' events';
      document.getElementById('last-update').textContent = new Date().toLocaleTimeString();

      renderEvents(data.events);

      const banner = document.getElementById('alert-banner');
      const recentViolations = (data.events || []).filter(e => e.event_type === 'PPE_NONCOMPLIANT_CROSSING');
      if (recentViolations.length > 0) {
        const latest = recentViolations[0];
        document.getElementById('alert-desc').textContent = `Non-compliant entry recorded on Camera ${latest.camera || '0'}.`;
        document.getElementById('alert-cam').textContent = `${latest.camera || '0'}`;
        document.getElementById('alert-time').textContent = formatTime(latest.timestamp_utc);
        banner.style.display = 'flex';
      } else {
        banner.style.display = 'none';
      }
    }

    // 2. Fetch Camera Status
    const respCams = await fetch('/api/cameras');
    if (respCams.ok) {
      const camJson = await respCams.json();
      cameraData = camJson.cameras || [];
      renderCameraCards();
    }
  } catch (err) {
    console.error('Refresh error:', err);
  }
}

refresh();
setInterval(refresh, 2500);
</script>
</body>
</html>"""


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves the dashboard HTML and JSON APIs for events and cameras."""

    csv_path: Path = Path("events.csv")

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/dashboard"):
            self._serve_html()
        elif parsed.path == "/api/events":
            self._serve_events_api()
        elif parsed.path == "/api/cameras":
            self._serve_cameras_api()
        elif parsed.path == "/api/cameras/scan":
            self._serve_cameras_scan()
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

    def _serve_events_api(self) -> None:
        events = read_events(self.csv_path)
        stats = compute_stats(events)
        payload = json.dumps({"events": events, "stats": stats}, ensure_ascii=False)
        self._send_json(payload)

    def _serve_cameras_api(self) -> None:
        summary = camera_manager.get_status_summary()
        self._send_json(json.dumps(summary, ensure_ascii=False))

    def _serve_cameras_scan(self) -> None:
        camera_manager.scan_cameras()
        summary = camera_manager.get_status_summary()
        self._send_json(json.dumps(summary, ensure_ascii=False))

    def _send_json(self, payload: str) -> None:
        content = payload.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="PPE Monitoring & CCTV Command Center")
    parser.add_argument("--events", default="events.csv", help="Path to the events CSV file")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port")
    args = parser.parse_args()

    DashboardHandler.csv_path = Path(args.events)
    print("=" * 65)
    print(" [PPE MONITORING & CCTV COMMAND CENTER]")
    print("=" * 65)
    print(f" Dashboard URL : http://localhost:{args.port}")
    print(f" Events Log    : {Path(args.events).resolve()}")
    print(" Initializing camera scan...")
    camera_manager.scan_cameras()
    print(f" Configured Cameras: {len(camera_manager.cameras)}")
    print(" Press Ctrl+C to stop.")
    print("=" * 65)

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
