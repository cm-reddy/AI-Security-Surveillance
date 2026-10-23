#!/usr/bin/env python3
"""Generate demo assets for PPE Webcam POC:
1. Synthetic demo events in events.csv to showcase the Admin Dashboard immediately.
2. Synthetic test video demo_video.mp4 with visual figures crossing a line to test video playback without webcam.
"""
from __future__ import annotations

import csv
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
import cv2
import numpy as np

CSV_FIELDS = [
    "timestamp_utc", "camera", "track_id", "event_type", "crossing_direction",
    "helmet_status", "vest_status", "overall_status", "model", "evidence_image",
]


def populate_demo_events(csv_path: Path = Path("events.csv"), count: int = 12) -> None:
    """Populate events.csv with realistic sample data for instant dashboard demo."""
    now = datetime.now(timezone.utc)
    scenarios = [
        ("0", 1, "PPE_COMPLIANT_CROSSING", "positive", "PRESENT", "PRESENT", "COMPLIANT"),
        ("0", 2, "PPE_NONCOMPLIANT_CROSSING", "positive", "MISSING", "PRESENT", "NON_COMPLIANT"),
        ("1", 3, "PPE_COMPLIANT_CROSSING", "positive", "PRESENT", "PRESENT", "COMPLIANT"),
        ("0", 4, "PPE_NONCOMPLIANT_CROSSING", "positive", "PRESENT", "MISSING", "NON_COMPLIANT"),
        ("1", 5, "PPE_UNKNOWN_AT_CROSSING", "positive", "UNKNOWN", "PRESENT", "UNKNOWN"),
        ("0", 6, "PPE_COMPLIANT_CROSSING", "positive", "PRESENT", "PRESENT", "COMPLIANT"),
        ("2", 7, "PPE_NONCOMPLIANT_CROSSING", "positive", "MISSING", "MISSING", "NON_COMPLIANT"),
        ("0", 8, "PPE_COMPLIANT_CROSSING", "positive", "PRESENT", "PRESENT", "COMPLIANT"),
        ("1", 9, "PPE_COMPLIANT_CROSSING", "positive", "PRESENT", "PRESENT", "COMPLIANT"),
        ("0", 10, "PPE_NONCOMPLIANT_CROSSING", "positive", "MISSING", "PRESENT", "NON_COMPLIANT"),
    ]

    rows = []
    for i, s in enumerate(scenarios[:count]):
        ts = (now - timedelta(minutes=(len(scenarios) - i) * 3 + 1)).isoformat(timespec="seconds")
        rows.append({
            "timestamp_utc": ts,
            "camera": s[0],
            "track_id": s[1],
            "event_type": s[2],
            "crossing_direction": s[3],
            "helmet_status": s[4],
            "vest_status": s[5],
            "overall_status": s[6],
            "model": "Hansung-Cho/yolov8-ppe-detection",
            "evidence_image": "",
        })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} sample crossing events in {csv_path.resolve()}")


def create_demo_video(video_path: Path = Path("demo_video.mp4"), duration_sec: int = 10, fps: int = 25) -> None:
    """Generate a lightweight synthetic demo video file for testing pipeline without webcam."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    line_y = int(height * 0.72)

    for frame_idx in range(total_frames):
        # Create dark industrial background
        frame = np.full((height, width, 3), (35, 30, 28), dtype=np.uint8)

        # Draw dual vertical yellow safety borders around machine cell
        left_x = int(width * 0.25)
        right_x = int(width * 0.75)
        border_col = (0, 220, 255)

        cv2.line(frame, (left_x, 50), (left_x, height - 10), border_col, 3)
        cv2.line(frame, (right_x, 50), (right_x, height - 10), border_col, 3)
        cv2.line(frame, (left_x, height - 10), (right_x, height - 10), border_col, 3)

        # Hazard stripes along vertical borders
        for y in range(70, height - 20, 35):
            cv2.line(frame, (left_x, y), (left_x + 12, y + 8), border_col, 2)
            cv2.line(frame, (right_x, y), (right_x - 12, y + 8), border_col, 2)

        # Labels
        cv2.putText(frame, "WALKWAY", (left_x // 2 - 35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 210, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, "WALKWAY", ((right_x + width) // 2 - 35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 210, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, "MACHINE ZONE - PPE REQUIRED", (left_x + 35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

        # Worker: walks from Left Walkway across left yellow border into Machine Zone
        progress = frame_idx / total_frames
        worker_x = int(left_x - 80 + progress * 240)
        worker_y = int(height * 0.55)

        # Draw simulated person figure
        # Head
        cv2.circle(frame, (worker_x, worker_y), 22, (200, 200, 200), -1)
        # Hardhat (yellow)
        cv2.ellipse(frame, (worker_x, worker_y - 8), (24, 12), 0, 180, 360, (0, 220, 255), -1)
        # Body / Safety vest (neon green)
        cv2.rectangle(frame, (worker_x - 30, worker_y + 22), (worker_x + 30, worker_y + 110), (0, 230, 100), -1)
        # Legs
        cv2.line(frame, (worker_x - 15, worker_y + 110), (worker_x - 15, worker_y + 170), (120, 120, 120), 8)
        cv2.line(frame, (worker_x + 15, worker_y + 110), (worker_x + 15, worker_y + 170), (120, 120, 120), 8)

        # Worker status label
        in_cell = worker_x >= left_x
        status_lbl = "Inside Machine Zone (PPE Verified)" if in_cell else "On Walkway (Approaching Border)"
        lbl_col = (0, 255, 120) if in_cell else (200, 200, 200)
        cv2.putText(frame, status_lbl, (max(10, worker_x - 110), worker_y - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, lbl_col, 1, cv2.LINE_AA)

        # OSD
        cv2.putText(frame, f"Machine Zone Simulation | Frame {frame_idx}/{total_frames}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"Generated synthetic demo video at {video_path.resolve()} ({duration_sec}s, {fps}fps)")


if __name__ == "__main__":
    populate_demo_events()
    create_demo_video()
