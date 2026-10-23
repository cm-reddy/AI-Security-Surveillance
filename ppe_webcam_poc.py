#!/usr/bin/env python3
"""Webcam PPE monitoring proof of concept.

This is a demonstration/monitoring aid, not a safety-rated access-control system.
It intentionally returns UNKNOWN when PPE evidence is missing or conflicting.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import re
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple
import webbrowser

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import winsound  # Windows only
except ImportError:  # pragma: no cover - depends on operating system
    winsound = None

DEFAULT_HF_MODEL = "hf:Hansung-Cho/yolov8-ppe-detection:best.pt"
CSV_FIELDS = [
    "timestamp_utc", "camera", "track_id", "event_type", "crossing_direction",
    "helmet_status", "vest_status", "overall_status", "model", "evidence_image",
]

# Status strings used throughout the system.
PRESENT = "PRESENT"
MISSING = "MISSING"
UNKNOWN = "UNKNOWN"
COMPLIANT = "COMPLIANT"
NON_COMPLIANT = "NON_COMPLIANT"


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def classify_label(label: str) -> str:
    """Map different PPE checkpoint model class names to semantic roles.

    Recognizes common aliases such as Hardhat, No-Hardhat, Safety Vest,
    No-Safety Vest, reflective_vest, and Person. Unrecognized classes are ignored.
    """
    n = normalize_label(label)

    # Person role first. This is a detector class, not a PPE status.
    if n in {"person", "people", "worker", "workers", "human"} or "person" == n:
        return "person"

    helmet_term = any(term in n for term in ("helmet", "hardhat"))
    vest_term = any(term in n for term in ("safetyvest", "reflectivevest", "highvisibilityvest", "highvisvest", "hi vis vest", "vest"))
    negative = (
        n.startswith(("no", "without", "missing", "unworn", "notwearing"))
        or "missing" in n
        or "notwearing" in n
        or "without" in n
    )

    if helmet_term:
        return "helmet_missing" if negative else "helmet_present"
    if vest_term:
        return "vest_missing" if negative else "vest_present"
    return "other"


@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    label: str
    confidence: float
    role: str
    track_id: Optional[int] = None


@dataclass
class TrackState:
    last_seen: float = field(default_factory=time.monotonic)
    last_stable_side: Optional[int] = None
    last_zone_state: Optional[str] = None
    last_crossing_at: float = 0.0
    last_violation_logged_at: float = 0.0
    helmet_history: Deque[str] = field(default_factory=lambda: deque(maxlen=8))
    vest_history: Deque[str] = field(default_factory=lambda: deque(maxlen=8))


def load_model(model_arg: str, world_classes: Optional[Sequence[str]] = None) -> Tuple[Any, str]:
    """Load a local/Ultralytics model, YOLO-World VLM, or download the default PPE fine-tune."""
    if "world" in model_arg.lower():
        from ultralytics import YOLOWorld
        weight_name = model_arg if model_arg.endswith(".pt") else "yolov8s-worldv2.pt"
        model = YOLOWorld(weight_name)
        classes = list(world_classes) if world_classes else ["person", "hardhat", "safety vest", "no hardhat", "no safety vest"]
        model.set_classes(classes)
        return model, f"YOLO-World ({weight_name})"

    if model_arg.startswith("hf:"):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required for hf: model references. Install requirements.txt."
            ) from exc
        spec = model_arg[3:]
        if ":" not in spec:
            raise ValueError("Hugging Face model syntax must be hf:owner/repo:filename")
        repo_id, filename = spec.rsplit(":", 1)
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return YOLO(model_path), str(model_path)

    # Ultralytics can load a local .pt path or its official model aliases,
    # downloading official weights if needed.
    model = YOLO(model_arg)
    return model, model_arg


def get_model_names(model: Any) -> Dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(name) for i, name in enumerate(names)}


def model_role_summary(names: Dict[int, str]) -> Dict[str, bool]:
    roles = {classify_label(name) for name in names.values()}
    return {
        "person": "person" in roles,
        "helmet_present": "helmet_present" in roles,
        "helmet_missing": "helmet_missing" in roles,
        "vest_present": "vest_present" in roles,
        "vest_missing": "vest_missing" in roles,
    }


def box_area(box: Tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def region_for_person(
    person_box: Tuple[float, float, float, float], item: str
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = person_box
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    if item == "helmet":
        # Head/helmet zone: upper central portion of the person box.
        return (x1 + 0.18 * w, y1, x2 - 0.18 * w, y1 + 0.36 * h)
    # Vest/torso zone: central upper body.
    return (x1 + 0.10 * w, y1 + 0.20 * h, x2 - 0.10 * w, y1 + 0.78 * h)


def overlap_fraction(
    box: Tuple[float, float, float, float], region: Tuple[float, float, float, float]
) -> float:
    ix1, iy1 = max(box[0], region[0]), max(box[1], region[1])
    ix2, iy2 = min(box[2], region[2]), min(box[3], region[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return intersection / max(1.0, box_area(box))


def center_in_region(
    box: Tuple[float, float, float, float], region: Tuple[float, float, float, float]
) -> bool:
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def associate_ppe(
    people: Sequence[Detection], detections: Sequence[Detection]
) -> Dict[int, Dict[str, List[Tuple[str, float]]]]:
    """Associate every PPE detection with at most one person's head/torso zone.

    Person list index is used as the internal key for the current frame. Each PPE
    object is assigned to the best-overlapping compatible region to reduce cases
    where one worker's PPE is credited to a neighboring worker.
    """
    evidence: Dict[int, Dict[str, List[Tuple[str, float]]]] = {
        i: {"helmet": [], "vest": []} for i in range(len(people))
    }

    for det in detections:
        if det.role not in {"helmet_present", "helmet_missing", "vest_present", "vest_missing"}:
            continue
        item = "helmet" if det.role.startswith("helmet") else "vest"
        candidates = []
        for person_idx, person in enumerate(people):
            region = region_for_person(person.xyxy, item)
            frac = overlap_fraction(det.xyxy, region)
            inside = center_in_region(det.xyxy, region)
            # Use either center containment or meaningful overlap. The scoring
            # chooses a single best wearer when adjacent regions overlap.
            if inside or frac >= 0.15:
                score = (2.0 if inside else 0.0) + frac
                candidates.append((score, person_idx))
        if not candidates:
            continue
        _, best_idx = max(candidates, key=lambda x: x[0])
        if det.role.endswith("missing"):
            status = MISSING
        else:
            status = PRESENT
        evidence[best_idx][item].append((status, det.confidence))
    return evidence


def frame_item_status(candidates: Sequence[Tuple[str, float]]) -> str:
    """Return a frame-level state; conflicting visible evidence remains UNKNOWN."""
    if not candidates:
        return UNKNOWN
    positives = [conf for state, conf in candidates if state == PRESENT]
    negatives = [conf for state, conf in candidates if state == MISSING]
    if positives and negatives:
        # Do not make a safety decision when this frame contradicts itself.
        if max(positives) >= 1.5 * max(negatives):
            return PRESENT
        if max(negatives) >= 1.5 * max(positives):
            return MISSING
        return UNKNOWN
    if positives:
        return PRESENT
    if negatives:
        return MISSING
    return UNKNOWN


def smooth_status(history: Deque[str], min_votes: int = 3) -> str:
    """Require repeated consistent evidence; absence of detections is UNKNOWN."""
    positives = sum(v == PRESENT for v in history)
    negatives = sum(v == MISSING for v in history)
    if positives >= min_votes and positives >= 2 * max(1, negatives):
        return PRESENT
    if negatives >= min_votes and negatives >= 2 * max(1, positives):
        return MISSING
    return UNKNOWN


def overall_status(helmet_status: str, vest_status: str) -> str:
    if helmet_status == MISSING or vest_status == MISSING:
        return NON_COMPLIANT
    if helmet_status == PRESENT and vest_status == PRESENT:
        return COMPLIANT
    return UNKNOWN


def person_anchor(person_box: Tuple[float, float, float, float], anchor: str) -> Tuple[int, int]:
    x1, y1, x2, y2 = person_box
    x = int((x1 + x2) / 2.0)
    y = int(y2 if anchor == "foot" else (y1 + y2) / 2.0)
    return x, y


def point_side(
    point: Tuple[int, int], frame_width: int, frame_height: int,
    orientation: str, position: float, hysteresis_px: int,
) -> int:
    coordinate = point[1] if orientation == "horizontal" else point[0]
    dimension = frame_height if orientation == "horizontal" else frame_width
    line_coordinate = int(dimension * position)
    delta = coordinate - line_coordinate
    if delta < -hysteresis_px:
        return -1
    if delta > hysteresis_px:
        return 1
    return 0


def line_endpoints(
    frame_width: int, frame_height: int, orientation: str, position: float
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if orientation == "horizontal":
        y = int(frame_height * position)
        return (0, y), (frame_width - 1, y)
    x = int(frame_width * position)
    return (x, 0), (x, frame_height - 1)


def check_zone_state(
    point: Tuple[int, int], frame_width: int, left_pos: float, right_pos: float, hysteresis_px: int
) -> str:
    """Classify person location as LEFT_WALKWAY, RIGHT_WALKWAY, MACHINE_ZONE, or TRANSITION."""
    x = point[0]
    left_x = int(frame_width * left_pos)
    right_x = int(frame_width * right_pos)

    if x < (left_x - hysteresis_px):
        return "LEFT_WALKWAY"
    elif x > (right_x + hysteresis_px):
        return "RIGHT_WALKWAY"
    elif (left_x + hysteresis_px) <= x <= (right_x - hysteresis_px):
        return "MACHINE_ZONE"
    return "TRANSITION"


def draw_machine_boundaries(
    frame, frame_w: int, frame_h: int, left_pos: float, right_pos: float,
    is_alert: bool = False, active_drag: Optional[str] = None, hover_border: Optional[str] = None
) -> None:
    """Draw industrial safety yellow machine cell borders with live drag handles."""
    left_x = int(frame_w * left_pos)
    right_x = int(frame_w * right_pos)
    top_y = 54
    bottom_y = frame_h - 12

    # Flashing alert red vs industrial safety yellow vs active drag cyan
    default_color = (0, 0, 255) if is_alert else (0, 220, 255)
    left_color = (255, 255, 0) if (active_drag == "left" or hover_border == "left") else default_color
    right_color = (255, 255, 0) if (active_drag == "right" or hover_border == "right") else default_color

    left_thick = 4 if (is_alert or active_drag == "left" or hover_border == "left") else 3
    right_thick = 4 if (is_alert or active_drag == "right" or hover_border == "right") else 3

    # Vertical perimeter borders
    cv2.line(frame, (left_x, top_y), (left_x, bottom_y), left_color, left_thick)
    cv2.line(frame, (right_x, top_y), (right_x, bottom_y), right_color, right_thick)
    # Bottom connector threshold
    cv2.line(frame, (left_x, bottom_y), (right_x, bottom_y), default_color, 2 if not is_alert else 3)
    # Top threshold line
    cv2.line(frame, (left_x, top_y), (right_x, top_y), default_color, 1 if not is_alert else 2)

    # Hazard caution stripes along vertical borders
    stripe_interval = 40
    for y in range(top_y + 16, bottom_y - 16, stripe_interval):
        cv2.line(frame, (left_x, y), (left_x + 12, y + 10), left_color, 2)
        cv2.line(frame, (right_x, y), (right_x - 12, y + 10), right_color, 2)

    # Walkway indicator badges on the outer sides
    left_walk_x = max(10, (left_x - 85) // 2)
    draw_badge(frame, "WALKWAY", left_walk_x, top_y + 22, (35, 55, 35), (180, 230, 180), scale=0.40, thickness=1)
    right_walk_x = min(frame_w - 95, (right_x + frame_w - 85) // 2)
    draw_badge(frame, "WALKWAY", right_walk_x, top_y + 22, (35, 55, 35), (180, 230, 180), scale=0.40, thickness=1)

    # Center Machine Zone Header (clean ASCII, no emojis that render as ?)
    center_x = (left_x + right_x) // 2
    if is_alert:
        zone_title = "[!] MACHINE ZONE VIOLATION - NON-COMPLIANT"
        title_bg = (0, 0, 220)
    else:
        zone_title = "MACHINE SAFETY ZONE | PPE MANDATORY"
        title_bg = (0, 130, 180)
    (tw, _), _ = cv2.getTextSize(zone_title, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    draw_badge(frame, zone_title, max(left_x + 8, center_x - tw // 2 - 8), top_y + 22, title_bg, (255, 255, 255), scale=0.42, thickness=1)

    # Mid-height interactive drag handles with live percentage readouts
    mid_y = (top_y + bottom_y) // 2
    l_handle_text = f"<||> {left_pos*100:.0f}%"
    r_handle_text = f"<||> {right_pos*100:.0f}%"
    l_handle_bg = (0, 170, 255) if (active_drag == "left" or hover_border == "left") else (35, 35, 35)
    r_handle_bg = (0, 170, 255) if (active_drag == "right" or hover_border == "right") else (35, 35, 35)
    draw_badge(frame, l_handle_text, left_x - 42, mid_y, l_handle_bg, (255, 255, 255), scale=0.36, thickness=1)
    draw_badge(frame, r_handle_text, right_x - 42, mid_y, r_handle_bg, (255, 255, 255), scale=0.36, thickness=1)


def make_sure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class EventLogger:
    def __init__(self, csv_path: Path, snapshots: bool = False, snapshot_dir: Path = Path("event_snapshots")):
        self.csv_path = csv_path
        self.save_snapshots = snapshots
        self.snapshot_dir = snapshot_dir
        self.recent_events: Deque[Dict[str, Any]] = deque(maxlen=25)
        make_sure_parent(csv_path)
        if snapshots:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
        else:
            # Load existing events so in-camera drawer shows recent history on startup
            try:
                with csv_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        self.recent_events.append(r)
            except Exception:
                pass

    def log(
        self, camera: str, track_id: int, event_type: str, direction: str,
        helmet: str, vest: str, model_label: str, frame,
    ) -> Optional[str]:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        evidence_image = ""
        if self.save_snapshots:
            safe_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            safe_camera = re.sub(r"[^A-Za-z0-9_-]+", "_", camera).strip("_") or "camera"
            filename = f"{safe_stamp}_cam{safe_camera}_track{track_id}_{event_type.lower()}.jpg"
            path = self.snapshot_dir / filename
            if cv2.imwrite(str(path), frame):
                evidence_image = str(path)

        row = {
            "timestamp_utc": timestamp,
            "camera": camera,
            "track_id": track_id,
            "event_type": event_type,
            "crossing_direction": direction,
            "helmet_status": helmet,
            "vest_status": vest,
            "overall_status": overall_status(helmet, vest),
            "model": model_label,
            "evidence_image": evidence_image,
        }
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writerow(row)
            f.flush()
        self.recent_events.append(row)
        return evidence_image or None


def _generate_industrial_siren_wav(duration: float = 0.65, sample_rate: int = 44100) -> bytes:
    """Synthesize a piercing, high-volume industrial dual-tone siren in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        n_samples = int(sample_rate * duration)
        frames = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            # 6 alternating rapid pulses between 2400 Hz and 1100 Hz at peak volume
            cycle = int(t * 10) % 2
            freq = 2400 if cycle == 0 else 1100
            val = int(32000 * math.sin(2 * math.pi * freq * t))
            frames.extend(val.to_bytes(2, byteorder="little", signed=True))
        wf.writeframes(frames)
    return buf.getvalue()


_ALARM_WAV_BYTES: Optional[bytes] = None
try:
    _ALARM_WAV_BYTES = _generate_industrial_siren_wav()
except Exception:
    _ALARM_WAV_BYTES = None


def play_warning_beep() -> None:
    """Issue a loud, piercing, high-urgency audible alarm through the multimedia audio system."""
    def _alarm_worker():
        played = False
        if winsound is not None:
            # 1. Primary: High-amplitude industrial siren played through Windows Multimedia sound card
            if _ALARM_WAV_BYTES is not None:
                try:
                    winsound.PlaySound(_ALARM_WAV_BYTES, winsound.SND_MEMORY)
                    played = True
                except Exception:
                    pass
            # 2. Secondary fallback: Windows Critical Alert sound through primary speakers
            if not played:
                try:
                    winsound.MessageBeep(winsound.MB_ICONHAND)
                    played = True
                except Exception:
                    pass
            # 3. Tertiary fallback: High-frequency alternating PC beeps
            if not played:
                try:
                    winsound.Beep(2400, 250)
                    winsound.Beep(1200, 200)
                    played = True
                except Exception:
                    pass
        if not played:
            sys.stdout.write("\a\a\a")
            sys.stdout.flush()

    threading.Thread(target=_alarm_worker, daemon=True).start()


def list_available_cameras(max_check: int = 6) -> List[Tuple[int, int, int]]:
    """Probe for available cameras and return list of (index, width, height)."""
    found = []
    for idx in range(max_check):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            found.append((idx, w, h))
            cap.release()
    return found

def draw_badge(
    frame, text: str, x: int, y: int, bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255), scale: float = 0.44, thickness: int = 1,
) -> Tuple[int, int, int, int]:
    """Draw a clean, anti-aliased pill badge with border and padding."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 8, 4
    x1 = max(0, x)
    y1 = max(0, y - th - pad_y)
    x2 = min(frame.shape[1] - 1, x1 + tw + 2 * pad_x)
    y2 = min(frame.shape[0] - 1, y1 + th + 2 * pad_y + baseline)

    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x1 + pad_x, y1 + th + pad_y - 1), font, scale, text_color, thickness, cv2.LINE_AA)
    return x1, y1, x2, y2


def draw_events_drawer(
    frame, recent_events: Sequence[Dict[str, Any]], frame_w: int, frame_h: int, hover_close: bool = False
) -> Tuple[int, int, int, int]:
    """Draw a semi-transparent dark HUD drawer displaying recent events on the right side."""
    drawer_w = min(420, max(320, int(frame_w * 0.40)))
    dx1 = frame_w - drawer_w
    dy1 = 54
    dx2 = frame_w - 6
    dy2 = frame_h - 12
    if dx2 <= dx1 or dy2 <= dy1:
        return (0, 0, 0, 0)

    # Semi-transparent dark overlay background (alpha 0.88)
    sub = frame[dy1:dy2, dx1:dx2]
    overlay = np.full_like(sub, 18, dtype=np.uint8)
    cv2.addWeighted(overlay, 0.88, sub, 0.12, 0, sub)
    frame[dy1:dy2, dx1:dx2] = sub

    # Border outline
    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 180, 255), 1, cv2.LINE_AA)

    # Drawer Header
    cv2.rectangle(frame, (dx1, dy1), (dx2, dy1 + 36), (28, 28, 35), -1)
    cv2.putText(frame, "LIVE EVENT AUDIT LOG", (dx1 + 12, dy1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

    # Close button [X]
    close_btn_rect = (dx2 - 32, dy1 + 6, dx2 - 8, dy1 + 30)
    cb_bg = (0, 0, 200) if hover_close else (45, 45, 55)
    cv2.rectangle(frame, (close_btn_rect[0], close_btn_rect[1]), (close_btn_rect[2], close_btn_rect[3]), cb_bg, -1)
    cv2.rectangle(frame, (close_btn_rect[0], close_btn_rect[1]), (close_btn_rect[2], close_btn_rect[3]), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "X", (close_btn_rect[0] + 7, close_btn_rect[1] + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Render events (most recent first)
    events_list = list(reversed(recent_events))[:6]
    cur_y = dy1 + 46

    if not events_list:
        cv2.putText(frame, "No events recorded yet.", (dx1 + 16, cur_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, "Walk into the Machine Zone", (dx1 + 16, cur_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, "to trigger an alarm & audit event.", (dx1 + 16, cur_y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
        return close_btn_rect

    for ev in events_list:
        if cur_y + 60 > dy2:
            break
        card_h = 56
        status = ev.get("overall_status", "")
        event_type = ev.get("event_type", "")
        track_id = ev.get("track_id", "?")
        t_str = ev.get("timestamp_utc", "")
        if "T" in t_str:
            time_part = t_str.split("T")[1][:8]
        else:
            time_part = t_str[-8:] if len(t_str) >= 8 else t_str

        is_violation = ("NONCOMPLIANT" in event_type or status == "NON_COMPLIANT")
        is_compliant = ("COMPLIANT" in event_type and not is_violation)
        tag_color = (0, 0, 220) if is_violation else ((0, 180, 0) if is_compliant else (0, 165, 255))
        tag_label = "BREACH" if is_violation else ("COMPLIANT" if is_compliant else "UNKNOWN")

        # Container card
        cv2.rectangle(frame, (dx1 + 8, cur_y), (dx2 - 8, cur_y + card_h), (25, 25, 30), -1)
        cv2.rectangle(frame, (dx1 + 8, cur_y), (dx2 - 8, cur_y + card_h), (50, 50, 60), 1)
        # Status strip
        cv2.rectangle(frame, (dx1 + 8, cur_y), (dx1 + 12, cur_y + card_h), tag_color, -1)

        draw_badge(frame, tag_label, dx1 + 18, cur_y + 18, tag_color, (255, 255, 255), scale=0.34, thickness=1)
        header_text = f"Worker #{track_id} | {time_part} UTC"
        cv2.putText(frame, header_text, (dx1 + 92, cur_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)

        h_st = ev.get("helmet_status", "?")
        v_st = ev.get("vest_status", "?")
        loc = ev.get("crossing_direction", "zone")
        detail_str = f"H: {h_st[:4]} | V: {v_st[:4]} | {loc}"
        cv2.putText(frame, detail_str, (dx1 + 18, cur_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA)

        cur_y += card_h + 6

    footer_text = "Press 'E' or click [X] to close"
    cv2.putText(frame, footer_text, (dx1 + 12, dy2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
    return close_btn_rect


def parse_source(value: str):
    if value.isdigit():
        return int(value)
    return value


def draw_detection(frame, det: Detection) -> None:
    x1, y1, x2, y2 = (int(v) for v in det.xyxy)
    if det.role.endswith("missing"):
        color = (0, 0, 255)
    elif det.role.endswith("present"):
        color = (0, 190, 0)
    elif det.role == "person":
        color = (255, 180, 0)
    else:
        color = (155, 155, 155)
    label = f"{det.label} {det.confidence:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.putText(frame, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Webcam PPE monitoring POC. Not a certified safety system."
    )
    parser.add_argument("--camera", default="0", help="Webcam index (0) or video path")
    parser.add_argument(
        "--model", default=DEFAULT_HF_MODEL,
        help=("YOLO model path/alias. Default is a third-party PPE-finetuned model; "
              "use --model yolo11n.pt for general COCO weights (PPE then remains UNKNOWN)."),
    )
    parser.add_argument("--conf", type=float, default=0.30, help="Detection confidence threshold")
    parser.add_argument("--person-conf", type=float, default=0.45,
                        help="Confidence threshold specifically for person detections (filters out background false positives, default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--line-orientation", choices=["horizontal", "vertical"], default="horizontal")
    parser.add_argument("--line-position", type=float, default=0.72, help="Line position as fraction of frame dimension (0–1)")
    parser.add_argument("--entry-direction", choices=["positive", "negative"], default="positive",
                        help="positive: top-to-bottom for horizontal line / left-to-right for vertical line")
    parser.add_argument("--anchor", choices=["foot", "center"], default="foot",
                        help="foot is appropriate only when the full person/feet are visible; center is demo-only")
    parser.add_argument("--hysteresis-px", type=int, default=10, help="Ignore small movements inside this band around the line")
    parser.add_argument("--min-votes", type=int, default=3, help="Repeated matching PPE observations needed per item")
    parser.add_argument("--history-size", type=int, default=8, help="Temporal PPE evidence window")
    parser.add_argument("--track-timeout", type=float, default=3.0, help="Seconds before a lost track is discarded")
    parser.add_argument("--beep-cooldown", type=float, default=1.0, help="Minimum seconds between audible warning beeps")
    parser.add_argument("--events", default="events.csv", help="CSV event log path")
    parser.add_argument("--save-snapshots", action="store_true", help="Save event snapshots (off by default for privacy)")
    parser.add_argument("--snapshot-dir", default="event_snapshots", help="Directory for optional event snapshots")
    parser.add_argument("--mirror", action="store_true", help="Mirror webcam image for a selfie-style demo")
    parser.add_argument("--no-audio", action="store_true", help="Disable audible warning beeps")
    parser.add_argument("--show-detections", action="store_true", help="Draw individual PPE detection boxes")
    parser.add_argument("--window-width", type=int, default=1280, help="Display window width in pixels (window is also resizable by dragging)")
    parser.add_argument("--window-height", type=int, default=720, help="Display window height in pixels")
    parser.add_argument("--boundary-type", choices=["dual_vertical", "single_line"], default="dual_vertical",
                        help="Boundary style: dual_vertical (yellow machine zone borders on left & right) or single_line (original line)")
    parser.add_argument("--zone-left", type=float, default=0.25,
                        help="Left vertical yellow border position as fraction of width (0-1, default: 0.25)")
    parser.add_argument("--zone-right", type=float, default=0.75,
                        help="Right vertical yellow border position as fraction of width (0-1, default: 0.75)")
    parser.add_argument("--fullscreen", action="store_true", help="Open window in fullscreen mode")
    parser.add_argument("--auto-log-violations", action="store_true", default=True,
                        help="Automatically log non-compliant workers in camera view to events.csv (default: True)")
    parser.add_argument("--no-auto-log-violations", dest="auto_log_violations", action="store_false",
                        help="Only log events when workers cross the virtual entry line")
    parser.add_argument("--violation-interval", type=float, default=6.0,
                        help="Seconds between repeated automatic violation logs for the same continuous worker")
    parser.add_argument("--world-classes", nargs="+",
                        default=["person", "hardhat", "safety vest", "no hardhat", "no safety vest"],
                        help="Zero-shot text classes for YOLO-World (e.g. --world-classes person hardhat safety-vest)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="Scan and list all connected internal and external webcams, then exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_cameras:
        print("\nScanning for connected video capture devices...")
        cams = list_available_cameras()
        if not cams:
            print("No cameras detected. Ensure your external webcam is securely plugged in.")
        else:
            print(f"Found {len(cams)} video capture device(s):")
            for idx, w, h in cams:
                desc = "Primary / Built-in" if idx == 0 else "External Webcam"
                res_str = f" ({w}x{h})" if (w and h) else ""
                print(f"  --> Camera Index {idx}{res_str} - {desc}")
            print("\nTo launch with a specific camera, pass --camera <index>, e.g.:")
            print(f"  python ppe_webcam_poc.py --camera {cams[-1][0]} --model weights/my_custom_ppe.pt\n")
        return 0
    if args.boundary_type == "dual_vertical":
        if not (0.05 <= args.zone_left < args.zone_right <= 0.95):
            print("ERROR: --zone-left must be less than --zone-right, and both between 0.05 and 0.95.", file=sys.stderr)
            return 2
    else:
        if not 0.05 <= args.line_position <= 0.95:
            print("ERROR: --line-position must be between 0.05 and 0.95.", file=sys.stderr)
            return 2
    if args.history_size < args.min_votes or args.min_votes < 1:
        print("ERROR: history-size must be >= min-votes >= 1.", file=sys.stderr)
        return 2

    print("Loading YOLO model...")
    try:
        model, model_label = load_model(args.model, args.world_classes)
    except Exception as exc:
        print(f"ERROR loading model: {exc}", file=sys.stderr)
        return 2

    names = get_model_names(model)
    role_summary = model_role_summary(names)
    print(f"Model: {model_label}")
    print("Model classes:", ", ".join(f"{i}:{name}" for i, name in names.items()))
    print("Required role coverage:")
    for role, available in role_summary.items():
        print(f"  {role}: {'FOUND' if available else 'NOT FOUND'}")
    if not role_summary["person"]:
        print("WARNING: This model has no recognizable person class. Person-specific compliance and crossing detection will not work.")
    if not role_summary["helmet_present"] or not role_summary["vest_present"]:
        print("WARNING: One or both positive PPE classes are missing. Those item statuses will stay UNKNOWN unless a recognized class is detected.")
    if not role_summary["helmet_missing"] or not role_summary["vest_missing"]:
        print("WARNING: One or both explicit missing-PPE classes are absent. No detection is NOT proof of missing PPE; those item statuses will stay UNKNOWN.")

    source = parse_source(args.camera)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera/video source {args.camera!r}. Close other camera apps or try --camera 1.", file=sys.stderr)
        return 2
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    logger = EventLogger(Path(args.events), snapshots=args.save_snapshots, snapshot_dir=Path(args.snapshot_dir))
    track_states: Dict[int, TrackState] = {}
    last_beep = 0.0
    last_frame_time = time.monotonic()
    smoothed_fps = 0.0
    camera_label = str(args.camera)
    last_violation_time = 0.0
    last_violation_msg = ""
    print("POC running. Press Q or Esc to quit. Use a well-lit view with the person's head, torso, and (for foot-anchor crossing) feet visible.")
    print("Important: warnings do not physically stop entry; do not use this as the sole safety barrier.")

    window_name = "Factory PPE Monitoring POC"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, args.window_width, args.window_height)

    # Dynamic interactive boundary settings (adjustable live via mouse or keyboard)
    boundary_state = {
        "boundary_type": args.boundary_type,
        "zone_left": float(args.zone_left),
        "zone_right": float(args.zone_right),
        "line_orientation": args.line_orientation,
        "line_position": float(args.line_position),
        "dragging": None,       # "left", "right", "line"
        "hover": None,          # "left", "right", "line"
        "frame_w": 1,
        "frame_h": 1,
        "status_notice": "",
        "status_notice_time": 0.0,
    }

    # UI State for interactive buttons and slide-out events drawer
    ui_state = {
        "show_events_drawer": False,
        "events_btn_rect": (0, 0, 0, 0),
        "dash_btn_rect": (0, 0, 0, 0),
        "close_btn_rect": (0, 0, 0, 0),
        "hover_btn": None,
    }

    def on_mouse(event, x, y, flags, param):
        fw = boundary_state["frame_w"]
        fh = boundary_state["frame_h"]
        if fw <= 1 or fh <= 1:
            return

        def in_rect(rx1, ry1, rx2, ry2):
            return rx1 <= x <= rx2 and ry1 <= y <= ry2

        # Check button hovers
        if event == cv2.EVENT_MOUSEMOVE:
            if in_rect(*ui_state["events_btn_rect"]):
                ui_state["hover_btn"] = "events"
            elif in_rect(*ui_state["dash_btn_rect"]):
                ui_state["hover_btn"] = "dashboard"
            elif ui_state["show_events_drawer"] and in_rect(*ui_state["close_btn_rect"]):
                ui_state["hover_btn"] = "close"
            else:
                ui_state["hover_btn"] = None

        # Check button clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            if in_rect(*ui_state["events_btn_rect"]):
                ui_state["show_events_drawer"] = not ui_state["show_events_drawer"]
                return
            elif in_rect(*ui_state["dash_btn_rect"]):
                webbrowser.open("http://localhost:8080")
                boundary_state["status_notice"] = "OPENING DASHBOARD (http://localhost:8080)..."
                boundary_state["status_notice_time"] = time.monotonic()
                return
            elif ui_state["show_events_drawer"] and in_rect(*ui_state["close_btn_rect"]):
                ui_state["show_events_drawer"] = False
                return
            elif ui_state["show_events_drawer"] and x >= (fw - 420):
                # Click inside drawer: consume event so borders don't drag
                return

        b_type = boundary_state["boundary_type"]
        if b_type == "dual_vertical":
            left_px = int(fw * boundary_state["zone_left"])
            right_px = int(fw * boundary_state["zone_right"])
            near_left = abs(x - left_px) <= 26
            near_right = abs(x - right_px) <= 26

            if event == cv2.EVENT_LBUTTONDOWN:
                if near_left:
                    boundary_state["dragging"] = "left"
                elif near_right:
                    boundary_state["dragging"] = "right"
            elif event == cv2.EVENT_MOUSEMOVE:
                if boundary_state["dragging"] == "left":
                    new_left = max(0.04, min(boundary_state["zone_right"] - 0.08, x / float(fw)))
                    boundary_state["zone_left"] = round(new_left, 3)
                    boundary_state["status_notice"] = f"LEFT BORDER: {boundary_state['zone_left']*100:.1f}%"
                    boundary_state["status_notice_time"] = time.monotonic()
                elif boundary_state["dragging"] == "right":
                    new_right = min(0.96, max(boundary_state["zone_left"] + 0.08, x / float(fw)))
                    boundary_state["zone_right"] = round(new_right, 3)
                    boundary_state["status_notice"] = f"RIGHT BORDER: {boundary_state['zone_right']*100:.1f}%"
                    boundary_state["status_notice_time"] = time.monotonic()
                else:
                    boundary_state["hover"] = "left" if near_left else ("right" if near_right else None)
            elif event == cv2.EVENT_LBUTTONUP:
                boundary_state["dragging"] = None
        else:
            orient = boundary_state["line_orientation"]
            dim = fh if orient == "horizontal" else fw
            coord = y if orient == "horizontal" else x
            line_px = int(dim * boundary_state["line_position"])
            near_line = abs(coord - line_px) <= 22

            if event == cv2.EVENT_LBUTTONDOWN:
                if near_line:
                    boundary_state["dragging"] = "line"
            elif event == cv2.EVENT_MOUSEMOVE:
                if boundary_state["dragging"] == "line":
                    new_pos = max(0.04, min(0.96, coord / float(dim)))
                    boundary_state["line_position"] = round(new_pos, 3)
                    boundary_state["status_notice"] = f"LINE: {boundary_state['line_position']*100:.1f}%"
                    boundary_state["status_notice_time"] = time.monotonic()
                else:
                    boundary_state["hover"] = "line" if near_line else None
            elif event == cv2.EVENT_LBUTTONUP:
                boundary_state["dragging"] = None

    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera/video frame unavailable; stopping the POC loop.", file=sys.stderr)
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            boundary_state["frame_w"] = frame_w
            boundary_state["frame_h"] = frame_h
            now = time.monotonic()

            try:
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )
            except Exception as exc:
                cv2.putText(frame, "INFERENCE ERROR - PPE STATUS UNKNOWN", (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)
                print(f"Inference error: {exc}", file=sys.stderr)
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            result = results[0]
            detections: List[Detection] = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxys = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                ids = result.boxes.id
                ids_list = ids.int().cpu().tolist() if ids is not None else [None] * len(classes)
                for coords, class_id, confidence, track_id in zip(xyxys, classes, confidences, ids_list):
                    label = names.get(int(class_id), f"class_{class_id}")
                    detections.append(Detection(
                        tuple(float(v) for v in coords), label, float(confidence),
                        classify_label(label), int(track_id) if track_id is not None else None,
                    ))

            # Filter person detections using dedicated person confidence threshold to eliminate false positives
            people = [d for d in detections if d.role == "person" and d.confidence >= args.person_conf]
            associated = associate_ppe(people, detections)
            visible_track_ids = set()
            crossing_texts: List[str] = []

            if args.show_detections:
                for det in detections:
                    draw_detection(frame, det)

            for idx, person in enumerate(people):
                person_track_id = person.track_id
                if person_track_id is None:
                    x1, y1, x2, y2 = (int(v) for v in person.xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 1)
                    cv2.putText(frame, "Acquiring Track...", (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)
                    continue

                visible_track_ids.add(person_track_id)
                state = track_states.get(person_track_id)
                if state is None:
                    state = TrackState(
                        helmet_history=deque(maxlen=args.history_size),
                        vest_history=deque(maxlen=args.history_size),
                    )
                    track_states[person_track_id] = state
                state.last_seen = now

                frame_evidence = associated.get(idx, {"helmet": [], "vest": []})
                state.helmet_history.append(frame_item_status(frame_evidence["helmet"]))
                state.vest_history.append(frame_item_status(frame_evidence["vest"]))
                helmet_status = smooth_status(state.helmet_history, min_votes=args.min_votes)
                vest_status = smooth_status(state.vest_history, min_votes=args.min_votes)
                overall = overall_status(helmet_status, vest_status)

                color = (0, 190, 0) if overall == COMPLIANT else ((0, 0, 255) if overall == NON_COMPLIANT else (0, 215, 255))
                x1, y1, x2, y2 = (int(v) for v in person.xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if overall != NON_COMPLIANT else 3)

                anchor_point = person_anchor(person.xyxy, args.anchor)

                # Determine zone label using dynamic interactive boundaries
                zone_tag = ""
                cur_zone = None
                if boundary_state["boundary_type"] == "dual_vertical":
                    cur_zone = check_zone_state(anchor_point, frame_w, boundary_state["zone_left"],
                                                boundary_state["zone_right"], args.hysteresis_px)
                    if cur_zone == "MACHINE_ZONE":
                        zone_tag = " [MACHINE]"
                    elif cur_zone in ("LEFT_WALKWAY", "RIGHT_WALKWAY"):
                        zone_tag = " [WALKWAY]"

                # Position badge above head if space exists, otherwise just inside person box to prevent overlapping
                badge_text = f"ID #{person_track_id}{zone_tag} | {overall}"
                badge_y = (y1 - 6) if y1 >= 90 else (y1 + 22)
                draw_badge(frame, badge_text, x1, badge_y, color, scale=0.44, thickness=1)

                detail_text = f"Helmet: {helmet_status} | Vest: {vest_status}"
                draw_badge(frame, detail_text, x1, min(frame_h - 10, y2 + 20), (35, 35, 35), (230, 230, 230), scale=0.38, thickness=1)

                # Active Machine Zone Monitoring & Presence Violation Logging
                if args.auto_log_violations and overall == NON_COMPLIANT:
                    in_danger_zone = (boundary_state["boundary_type"] != "dual_vertical") or (cur_zone == "MACHINE_ZONE")
                    if in_danger_zone:
                        # Loud audible alarm fires immediately & repeatedly while in danger zone
                        if not args.no_audio and (now - last_beep) >= args.beep_cooldown:
                            play_warning_beep()
                            last_beep = now

                        # Rate-limit CSV logging so events.csv doesn't get flooded every frame
                        if (now - state.last_violation_logged_at) >= args.violation_interval:
                            state.last_violation_logged_at = now
                            event_type = "PPE_NONCOMPLIANT_CROSSING"
                            loc_desc = "machine_zone" if (boundary_state["boundary_type"] == "dual_vertical" and cur_zone == "MACHINE_ZONE") else "camera_view"
                            logger.log(camera_label, person_track_id, event_type, loc_desc,
                                       helmet_status, vest_status, model_label, frame)
                            crossing_texts.append(f"ID #{person_track_id}: VIOLATION LOGGED")
                            last_violation_time = now
                            last_violation_msg = f"WORKER #{person_track_id} IN MACHINE ZONE WITHOUT PPE"
                            print(f"[ALERT] Logged PPE violation for Worker #{person_track_id} ({loc_desc}) to {args.events}")

                # Boundary crossing detection (dual vertical machine borders or single entry line)
                if boundary_state["boundary_type"] == "dual_vertical":
                    zone = cur_zone or check_zone_state(anchor_point, frame_w, boundary_state["zone_left"],
                                                        boundary_state["zone_right"], args.hysteresis_px)
                    if zone in ("LEFT_WALKWAY", "RIGHT_WALKWAY", "MACHINE_ZONE"):
                        prev_zone = state.last_zone_state
                        if prev_zone is not None and zone != prev_zone:
                            # Worker crossed boundary from walkway into the machine zone
                            if prev_zone in ("LEFT_WALKWAY", "RIGHT_WALKWAY") and zone == "MACHINE_ZONE":
                                direction = "entry_from_left" if prev_zone == "LEFT_WALKWAY" else "entry_from_right"
                                if (now - state.last_crossing_at) >= 0.25:
                                    state.last_crossing_at = now
                                    if overall == NON_COMPLIANT:
                                        event_type = "PPE_NONCOMPLIANT_CROSSING"
                                        last_violation_time = now
                                        last_violation_msg = f"WORKER #{person_track_id} BREACHED MACHINE CELL ({direction.upper()})"
                                    elif overall == UNKNOWN:
                                        event_type = "PPE_UNKNOWN_AT_CROSSING"
                                        last_violation_time = now
                                        last_violation_msg = f"WORKER #{person_track_id} ENTERED MACHINE CELL (UNVERIFIED)"
                                    else:
                                        event_type = "PPE_COMPLIANT_CROSSING"
                                    logger.log(camera_label, person_track_id, event_type, direction,
                                               helmet_status, vest_status, model_label, frame)
                                    crossing_texts.append(f"ID #{person_track_id}: {direction} -> {event_type}")
                                    print(f"[ZONE ENTRY] Worker #{person_track_id} crossed yellow border into Machine Zone ({direction}) -> {event_type}")
                                    if overall != COMPLIANT and not args.no_audio and (now - last_beep) >= args.beep_cooldown:
                                        play_warning_beep()
                                        last_beep = now
                            elif prev_zone == "MACHINE_ZONE" and zone in ("LEFT_WALKWAY", "RIGHT_WALKWAY"):
                                crossing_texts.append(f"ID #{person_track_id}: EXITED TO WALKWAY")
                        state.last_zone_state = zone
                else:
                    side = point_side(anchor_point, frame_w, frame_h, boundary_state["line_orientation"],
                                      boundary_state["line_position"], args.hysteresis_px)
                    if side != 0:
                        previous_side = state.last_stable_side
                        if previous_side is not None and side != previous_side:
                            direction = "positive" if previous_side == -1 and side == 1 else "negative"
                            is_entry_crossing = direction == args.entry_direction
                            if is_entry_crossing and (now - state.last_crossing_at) >= 0.25:
                                state.last_crossing_at = now
                                if overall == NON_COMPLIANT:
                                    event_type = "PPE_NONCOMPLIANT_CROSSING"
                                    last_violation_time = now
                                    last_violation_msg = f"WORKER #{person_track_id} - PPE NON-COMPLIANT"
                                elif overall == UNKNOWN:
                                    event_type = "PPE_UNKNOWN_AT_CROSSING"
                                    last_violation_time = now
                                    last_violation_msg = f"WORKER #{person_track_id} - PPE UNVERIFIED"
                                else:
                                    event_type = "PPE_COMPLIANT_CROSSING"
                                logger.log(camera_label, person_track_id, event_type, direction,
                                           helmet_status, vest_status, model_label, frame)
                                crossing_texts.append(f"ID {person_track_id}: {event_type}")
                                if overall != COMPLIANT and not args.no_audio and (now - last_beep) >= args.beep_cooldown:
                                    play_warning_beep()
                                    last_beep = now
                        state.last_stable_side = side

                cv2.circle(frame, anchor_point, 5, color, -1)

            # Remove old track state so IDs do not accumulate forever.
            expired = [tid for tid, state in track_states.items() if now - state.last_seen > args.track_timeout]
            for tid in expired:
                del track_states[tid]

            # Render boundaries: Dual vertical yellow machine borders OR single line
            if boundary_state["boundary_type"] == "dual_vertical":
                is_breach = (now - last_violation_time) < 3.5
                draw_machine_boundaries(frame, frame_w, frame_h,
                                        boundary_state["zone_left"], boundary_state["zone_right"],
                                        is_breach, boundary_state["dragging"], boundary_state["hover"])
            else:
                p1, p2 = line_endpoints(frame_w, frame_h, boundary_state["line_orientation"], boundary_state["line_position"])
                line_col = (255, 255, 0) if (boundary_state["dragging"] == "line" or boundary_state["hover"] == "line") else (255, 180, 0)
                cv2.line(frame, p1, p2, line_col, 3 if boundary_state["dragging"] == "line" else 2)
                mid_x, mid_y = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                handle_lbl = f"< DRAG LINE > {boundary_state['line_position']*100:.0f}%"
                draw_badge(frame, handle_lbl, mid_x - 65, mid_y - 10,
                           (0, 160, 255) if boundary_state["dragging"] == "line" else (35, 35, 35),
                           (255, 255, 255), scale=0.38, thickness=1)

            dt = max(1e-6, now - last_frame_time)
            instantaneous_fps = 1.0 / dt
            smoothed_fps = instantaneous_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instantaneous_fps
            last_frame_time = now

            # Top HUD Bar (height 52px)
            cv2.rectangle(frame, (0, 0), (frame_w, 52), (20, 20, 20), -1)
            cv2.line(frame, (0, 52), (frame_w, 52), (50, 50, 50), 1)

            cv2.putText(frame, "SAFETY ADVISORY: MONITORING AID ONLY (DOES NOT PREVENT ENTRY)", (12, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 165, 255), 1, cv2.LINE_AA)

            hud_line2_left = f"FPS: {smoothed_fps:.1f} | Workers: {len(people)}"
            if boundary_state["boundary_type"] == "dual_vertical":
                hud_line2_left += f" | Machine: {boundary_state['zone_left']*100:.0f}%-{boundary_state['zone_right']*100:.0f}%"
            cv2.putText(frame, hud_line2_left, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

            # Draw Interactive Top-Right Buttons: [ 🌐 Dashboard (D) ] and [ 📋 Events (E) (N) ]
            btn_y1 = 8
            btn_y2 = 44

            events_count = len(logger.recent_events)
            events_text = f"📋 Events ({events_count})" if not ui_state["show_events_drawer"] else "📋 Events [X]"
            (etw, _), _ = cv2.getTextSize(events_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            eb_w = etw + 22
            eb_x2 = frame_w - 12
            eb_x1 = eb_x2 - eb_w
            ui_state["events_btn_rect"] = (eb_x1, btn_y1, eb_x2, btn_y2)

            dash_text = "🌐 Dashboard"
            (dtw, _), _ = cv2.getTextSize(dash_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            db_w = dtw + 20
            db_x2 = eb_x1 - 10
            db_x1 = db_x2 - db_w
            ui_state["dash_btn_rect"] = (db_x1, btn_y1, db_x2, btn_y2)

            # Events button styling
            eb_bg = (0, 180, 255) if ui_state["show_events_drawer"] else ((0, 140, 200) if ui_state["hover_btn"] == "events" else (40, 40, 50))
            eb_text_col = (0, 0, 0) if ui_state["show_events_drawer"] else (255, 255, 255)
            cv2.rectangle(frame, (eb_x1, btn_y1), (eb_x2, btn_y2), eb_bg, -1)
            cv2.rectangle(frame, (eb_x1, btn_y1), (eb_x2, btn_y2), (0, 220, 255) if ui_state["show_events_drawer"] else (80, 80, 90), 1, cv2.LINE_AA)
            cv2.putText(frame, events_text, (eb_x1 + 10, btn_y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.40, eb_text_col, 1, cv2.LINE_AA)

            # Dashboard button styling
            db_bg = (0, 170, 120) if ui_state["hover_btn"] == "dashboard" else (35, 45, 40)
            cv2.rectangle(frame, (db_x1, btn_y1), (db_x2, btn_y2), db_bg, -1)
            cv2.rectangle(frame, (db_x1, btn_y1), (db_x2, btn_y2), (0, 200, 140), 1, cv2.LINE_AA)
            cv2.putText(frame, dash_text, (db_x1 + 10, btn_y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

            # Central adjustment notice if active
            if (now - boundary_state["status_notice_time"]) < 2.5 and boundary_state["status_notice"]:
                notice_text = f"⚙ {boundary_state['status_notice']}"
                (nw, _), _ = cv2.getTextSize(notice_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.putText(frame, notice_text, (max(160, db_x1 - nw - 14), 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

            # Bottom Warning Banner (Auto-scaled text so it never cuts off)
            if (now - last_violation_time) < 3.5:
                flash = int((now - last_violation_time) * 4) % 2 == 0
                banner_color = (0, 0, 230) if flash else (0, 0, 150)
                banner_h = 50
                by1 = frame_h - banner_h
                cv2.rectangle(frame, (0, by1), (frame_w, frame_h), banner_color, -1)
                cv2.rectangle(frame, (0, by1), (frame_w, frame_h), (255, 255, 255), 2)

                alert_text = f"[!] WARNING: {last_violation_msg} - MANDATORY PPE REQUIRED"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.58
                thick = 2
                (tw, th), _ = cv2.getTextSize(alert_text, font, font_scale, thick)
                # Auto scale down if text exceeds window width
                if tw > (frame_w - 40):
                    font_scale = max(0.38, font_scale * ((frame_w - 40) / float(tw)))
                    thick = 1
                    (tw, th), _ = cv2.getTextSize(alert_text, font, font_scale, thick)
                text_x = max(10, (frame_w - tw) // 2)
                text_y = by1 + (banner_h + th) // 2
                cv2.putText(frame, alert_text, (text_x, text_y), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
            elif crossing_texts:
                y = frame_h - 18
                for message in crossing_texts[-3:]:
                    cv2.putText(frame, message, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                (0, 0, 255), 1, cv2.LINE_AA)
                    y -= 20

            # Render Slide-Out Events Drawer if toggled open
            if ui_state["show_events_drawer"]:
                ui_state["close_btn_rect"] = draw_events_drawer(
                    frame, logger.recent_events, frame_w, frame_h, hover_close=(ui_state["hover_btn"] == "close")
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                # Interactive test trigger: immediately log event for currently detected worker
                tid = people[0].track_id if (people and people[0].track_id is not None) else 1
                st = track_states.get(tid)
                h_st = smooth_status(st.helmet_history) if st else MISSING
                v_st = smooth_status(st.vest_history) if st else MISSING
                ov = overall_status(h_st, v_st)
                ev = "PPE_NONCOMPLIANT_CROSSING" if ov != COMPLIANT else "PPE_COMPLIANT_CROSSING"
                logger.log(camera_label, tid, ev, "manual_test", h_st, v_st, model_label, frame)
                last_violation_time = now
                last_violation_msg = f"TEST EVENT: WORKER #{tid} ({ev})"
                crossing_texts.append(f"MANUAL TEST: {ev}")
                print(f"[MANUAL TEST] Logged {ev} for Track #{tid} to {args.events}")
                if not args.no_audio:
                    play_warning_beep()
            elif key in (ord("["), ord("{")):
                # Move left border left (or line down/left)
                if boundary_state["boundary_type"] == "dual_vertical":
                    boundary_state["zone_left"] = max(0.04, round(boundary_state["zone_left"] - 0.02, 3))
                    boundary_state["status_notice"] = f"LEFT BORDER: {boundary_state['zone_left']*100:.1f}%"
                else:
                    boundary_state["line_position"] = max(0.04, round(boundary_state["line_position"] - 0.02, 3))
                    boundary_state["status_notice"] = f"LINE: {boundary_state['line_position']*100:.1f}%"
                boundary_state["status_notice_time"] = now
            elif key in (ord("]"), ord("}")):
                # Move left border right (or line up/right)
                if boundary_state["boundary_type"] == "dual_vertical":
                    boundary_state["zone_left"] = min(boundary_state["zone_right"] - 0.08, round(boundary_state["zone_left"] + 0.02, 3))
                    boundary_state["status_notice"] = f"LEFT BORDER: {boundary_state['zone_left']*100:.1f}%"
                else:
                    boundary_state["line_position"] = min(0.96, round(boundary_state["line_position"] + 0.02, 3))
                    boundary_state["status_notice"] = f"LINE: {boundary_state['line_position']*100:.1f}%"
                boundary_state["status_notice_time"] = now
            elif key in (ord(","), ord("<")):
                # Move right border left
                if boundary_state["boundary_type"] == "dual_vertical":
                    boundary_state["zone_right"] = max(boundary_state["zone_left"] + 0.08, round(boundary_state["zone_right"] - 0.02, 3))
                    boundary_state["status_notice"] = f"RIGHT BORDER: {boundary_state['zone_right']*100:.1f}%"
                    boundary_state["status_notice_time"] = now
            elif key in (ord("."), ord(">")):
                # Move right border right
                if boundary_state["boundary_type"] == "dual_vertical":
                    boundary_state["zone_right"] = min(0.96, round(boundary_state["zone_right"] + 0.02, 3))
                    boundary_state["status_notice"] = f"RIGHT BORDER: {boundary_state['zone_right']*100:.1f}%"
                    boundary_state["status_notice_time"] = now
            elif key in (ord("t"), ord("T"), 9):  # 9 is TAB key
                boundary_state["boundary_type"] = "single_line" if boundary_state["boundary_type"] == "dual_vertical" else "dual_vertical"
                boundary_state["status_notice"] = f"MODE: {boundary_state['boundary_type'].upper()}"
                boundary_state["status_notice_time"] = now
            elif key in (ord("h"), ord("H")):
                boundary_state["line_orientation"] = "horizontal"
                boundary_state["status_notice"] = "LINE: HORIZONTAL"
                boundary_state["status_notice_time"] = now
            elif key in (ord("v"), ord("V")):
                boundary_state["line_orientation"] = "vertical"
                boundary_state["status_notice"] = "LINE: VERTICAL"
                boundary_state["status_notice_time"] = now
            elif key in (ord("r"), ord("R")):
                boundary_state["zone_left"] = float(args.zone_left)
                boundary_state["zone_right"] = float(args.zone_right)
                boundary_state["line_position"] = float(args.line_position)
                boundary_state["status_notice"] = "RESET TO DEFAULTS"
                boundary_state["status_notice_time"] = now
            elif key in (ord("e"), ord("E")):
                ui_state["show_events_drawer"] = not ui_state["show_events_drawer"]
            elif key in (ord("d"), ord("D")):
                webbrowser.open("http://localhost:8080")
                boundary_state["status_notice"] = "OPENING DASHBOARD..."
                boundary_state["status_notice_time"] = now

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Event log: {Path(args.events).resolve()}")
        if args.save_snapshots:
            print(f"Event snapshots: {Path(args.snapshot_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
