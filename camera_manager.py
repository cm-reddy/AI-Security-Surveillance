#!/usr/bin/env python3
"""Enterprise Camera Management and Stream Ingestion Layer.

Supports:
- Dedicated IP CCTV cameras (RTSP / HTTP / ONVIF feeds) with authentication
- Local USB webcams (0, 1, ...)
- Pre-recorded simulation feeds (MP4, MKV)
- Dynamic discovery & non-blocking health checks (fast socket ping)
- Resilient background auto-reconnect loop (never crashes on network drop)
- High-definition "NOT CONNECTED" visual fallback screen with countdown & diagnostic tips
- Zero-latency threaded frame buffer (drops stale frames to prevent RTSP buffer lag)
"""
from __future__ import annotations

import json
import logging
import math
import os
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

logger = logging.getLogger("CameraManager")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s")


@dataclass
class CameraInfo:
    id: str
    name: str
    type: str  # 'ip', 'usb', 'file'
    source: str
    location: str = "General"
    enabled: bool = True
    reconnect_interval_sec: float = 3.0
    description: str = ""
    status: str = "INITIALIZING"  # 'CONNECTED', 'NOT_CONNECTED', 'RECONNECTING', 'DISABLED'
    last_seen: float = 0.0
    fps: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    consecutive_failures: int = 0
    error_message: str = ""
    last_seen_str: str = "Never"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["last_seen_str"] = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_seen))
            if self.last_seen > 0
            else "Never"
        )
        return d


def fast_tcp_ping(host: str, port: int, timeout: float = 1.2) -> bool:
    """Test TCP connectivity without blocking OpenCV VideoCapture indefinitely."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def probe_source(source_str: str, timeout: float = 2.0) -> Tuple[bool, str, Tuple[int, int]]:
    """Quickly probe a camera source to see if it is currently reachable."""
    # 1. Numeric USB camera index
    if source_str.isdigit():
        idx = int(source_str)
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened() and sys.platform == "win32":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            return True, "USB device active", (w, h)
        return False, f"USB camera index {idx} not responding", (0, 0)

    # 2. Local video file
    path = Path(source_str)
    if path.exists() and path.is_file():
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            return True, "Video file accessible", (w, h)
        return False, f"Unable to decode video file {source_str}", (0, 0)

    # 3. Network IP camera (RTSP / HTTP)
    if "://" in source_str:
        parsed = urlparse(source_str)
        host = parsed.hostname
        port = parsed.port
        if not port:
            port = 554 if parsed.scheme.lower() == "rtsp" else 80

        if host:
            is_alive = fast_tcp_ping(host, port, timeout=1.0)
            if not is_alive:
                return False, f"Host {host}:{port} unreachable (ping failed)", (0, 0)

        # Socket reached, test VideoCapture open with brief timeout
        cap = cv2.VideoCapture(source_str)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                cap.release()
                return True, "Stream connected", (w, h)
            cap.release()
            return False, "Stream opened but failed to yield frame", (0, 0)
        return False, "Could not open RTSP/HTTP stream", (0, 0)

    return False, f"Unknown source format: {source_str}", (0, 0)


def render_disconnected_screen(
    camera_name: str,
    source: str,
    status_text: str,
    countdown_sec: float,
    attempt: int,
    width: int = 1280,
    height: int = 720,
) -> np.ndarray:
    """Generate a sleek, dark industrial fallback frame when camera is disconnected.
    
    Prevents GUI/stream crashes and clearly communicates connection health.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Dark modern slate gradient background
    frame[:, :] = (18, 14, 11)  # BGR #0B0E12

    # Subtle grid overlay
    grid_spacing = 40
    color_grid = (26, 22, 18)
    for x in range(0, width, grid_spacing):
        cv2.line(frame, (x, 0), (x, height), color_grid, 1)
    for y in range(0, height, grid_spacing):
        cv2.line(frame, (0, y), (width, y), color_grid, 1)

    # Warning Box in center
    bw, bh = min(780, width - 60), min(360, height - 60)
    bx1 = (width - bw) // 2
    by1 = (height - bh) // 2
    bx2 = bx1 + bw
    by2 = by1 + bh

    # Card background with glassmorphism dark tint
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (32, 28, 24), -1)
    
    # Pulsing red/amber border
    pulse = int(127 + 127 * math.sin(time.time() * 3.5))
    border_color = (40, 40, max(120, pulse))  # BGR Red glow
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), border_color, 2)

    # Title & Header
    title_y = by1 + 55
    cv2.putText(
        frame,
        "CAMERA NOT CONNECTED",
        (bx1 + 35, title_y),
        cv2.FONT_HERSHEY_DUPLEX,
        1.1,
        (60, 60, 245),
        2,
        cv2.LINE_AA,
    )

    # Status Pill
    badge_x = bx2 - 200
    cv2.rectangle(frame, (badge_x, by1 + 25), (badge_x + 165, by1 + 65), (20, 20, 160), -1)
    cv2.putText(
        frame,
        "OFFLINE",
        (badge_x + 35, by1 + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Horizontal separator
    cv2.line(frame, (bx1 + 35, title_y + 20), (bx2 - 35, title_y + 20), (55, 50, 45), 1)

    # Camera Details
    info_y = title_y + 55
    cv2.putText(
        frame,
        f"Camera Name : {camera_name}",
        (bx1 + 40, info_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    masked_source = source
    if "@" in masked_source:
        # Mask password in rtsp://user:pass@host
        prefix, rest = masked_source.split("://", 1)
        cred, host_part = rest.split("@", 1)
        masked_source = f"{prefix}://***:***@{host_part}"

    cv2.putText(
        frame,
        f"Endpoint    : {masked_source}",
        (bx1 + 40, info_y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Reason      : {status_text}",
        (bx1 + 40, info_y + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (120, 120, 240),
        1,
        cv2.LINE_AA,
    )

    # Reconnection status & countdown
    countdown_str = f"{max(0.0, countdown_sec):.1f}s"
    retry_msg = f"Automatic Reconnection: Attempt #{attempt} in {countdown_str}"
    cv2.putText(
        frame,
        retry_msg,
        (bx1 + 40, info_y + 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 215, 255),  # Gold/Amber
        2,
        cv2.LINE_AA,
    )

    # Loading dots animation
    dots = "." * ((int(time.time() * 2)) % 4)
    cv2.putText(
        frame,
        f"Listening for signal{dots}",
        (bx1 + 40, info_y + 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )

    # Diagnostics footer
    cv2.putText(
        frame,
        "Troubleshooting: Verify IP network, RTSP port 554, PoE power, or USB connection.",
        (bx1 + 40, by2 - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (130, 125, 120),
        1,
        cv2.LINE_AA,
    )

    # Bottom screen banner
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame,
        f"PPE SURVEILLANCE ENGINE | STATUS: RECONNECTING | {now_str}",
        (35, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (90, 85, 80),
        1,
        cv2.LINE_AA,
    )

    return frame


class ResilientCameraStream:
    """Zero-latency threaded camera reader with auto-reconnection and offline HUD."""

    def __init__(self, config: CameraInfo, width: int = 1280, height: int = 720):
        self.config = config
        self.target_width = width
        self.target_height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.latest_frame: Optional[np.ndarray] = None
        self.is_connected = False
        self.reconnect_attempt = 0
        self.next_reconnect_time = 0.0
        self.fps_tracker_time = time.monotonic()
        self.frame_count = 0

    def start(self):
        """Start the background ingestion thread."""
        if self.running:
            return
        # Eagerly attempt initial connection so frame 1 is immediately available
        self._try_open()
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True, name=f"CamThread-{self.config.id}")
        self.thread.start()
        logger.info(f"Started camera worker for '{self.config.name}' ({self.config.source})")

    def stop(self):
        """Stop worker and release camera handle."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self._release_cap()
        logger.info(f"Stopped camera worker for '{self.config.name}'")

    def _release_cap(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_connected = False

    def _try_open(self) -> bool:
        self._release_cap()
        source = self.config.source
        actual_source: Any = int(source) if source.isdigit() else source

        # Pre-check IP network connectivity to avoid 30s ffmpeg blocking
        if isinstance(actual_source, str) and "://" in actual_source:
            parsed = urlparse(actual_source)
            host = parsed.hostname
            port = parsed.port or (554 if parsed.scheme.lower() == "rtsp" else 80)
            if host and not fast_tcp_ping(host, port, timeout=0.8):
                self.config.status = "NOT_CONNECTED"
                self.config.error_message = f"Host {host}:{port} unreachable (ping failed)"
                return False
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|stimeout;2000000"
            )

        try:
            if isinstance(actual_source, int):
                cap = cv2.VideoCapture(actual_source)
                if not cap.isOpened() and sys.platform == "win32":
                    cap = cv2.VideoCapture(actual_source, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            else:
                cap = cv2.VideoCapture(actual_source)

            if cap.isOpened():
                ok, test_frame = cap.read()
                if ok and test_frame is not None:
                    self.cap = cap
                    self.is_connected = True
                    self.reconnect_attempt = 0
                    h, w = test_frame.shape[:2]
                    self.config.resolution = (w, h)
                    self.config.status = "CONNECTED"
                    self.config.error_message = ""
                    self.config.last_seen = time.time()
                    with self.lock:
                        self.latest_frame = test_frame.copy()
                    logger.info(f"Successfully connected to '{self.config.name}' ({w}x{h})")
                    return True
                else:
                    cap.release()
        except Exception as e:
            self.config.error_message = str(e)

        self.is_connected = False
        self.config.status = "NOT_CONNECTED"
        return False

    def _worker_loop(self):
        while self.running:
            if not self.is_connected:
                now = time.monotonic()
                if now >= self.next_reconnect_time:
                    self.reconnect_attempt += 1
                    self.config.status = "RECONNECTING"
                    success = self._try_open()
                    if not success:
                        # Exponential backoff up to 10s
                        backoff = min(10.0, self.config.reconnect_interval_sec * (1.2 ** min(self.reconnect_attempt, 5)))
                        self.next_reconnect_time = time.monotonic() + backoff
                        self.config.status = "NOT_CONNECTED"
                time.sleep(0.1)
                continue

            # Read frame continuously from stream
            if self.cap is not None and self.cap.isOpened():
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    self.config.last_seen = time.time()
                    self.frame_count += 1
                    elapsed = time.monotonic() - self.fps_tracker_time
                    if elapsed >= 2.0:
                        self.config.fps = round(self.frame_count / elapsed, 1)
                        self.frame_count = 0
                        self.fps_tracker_time = time.monotonic()

                    with self.lock:
                        self.latest_frame = frame
                else:
                    if self.config.type == "file":
                        # Seamlessly loop simulation video files
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.03)
                        continue

                    # Dropped frame / broken connection for live streams
                    logger.warning(f"Connection lost on '{self.config.name}'. Triggering reconnect loop.")
                    self._release_cap()
                    self.config.status = "NOT_CONNECTED"
                    self.config.error_message = "Stream interrupted or disconnected."
                    self.next_reconnect_time = time.monotonic() + self.config.reconnect_interval_sec
            else:
                self.is_connected = False
                time.sleep(0.1)

    def read(self) -> Tuple[bool, np.ndarray]:
        """Return (is_connected, frame). If disconnected, returns a synthetic offline frame."""
        with self.lock:
            if self.is_connected and self.latest_frame is not None:
                return True, self.latest_frame.copy()

        # Generate live disconnected screen
        countdown = max(0.0, self.next_reconnect_time - time.monotonic())
        offline_frame = render_disconnected_screen(
            camera_name=self.config.name,
            source=self.config.source,
            status_text=self.config.error_message or "Signal lost / Stream unreachable",
            countdown_sec=countdown,
            attempt=self.reconnect_attempt,
            width=self.target_width,
            height=self.target_height,
        )
        return False, offline_frame


class CameraManager:
    """Central registry and manager for all cameras (IP CCTV, USB, files)."""

    def __init__(self, config_path: str = "cameras.json"):
        self.config_path = Path(config_path).resolve()
        self.cameras: Dict[str, CameraInfo] = {}
        self.active_camera_id: str = "usb_0"
        self.active_stream: Optional[ResilientCameraStream] = None
        self._load_config()

    def _load_config(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.active_camera_id = data.get("active_camera_id", "usb_0")
                    valid_keys = set(CameraInfo.__dataclass_fields__.keys())
                    for c in data.get("cameras", []):
                        filtered = {k: v for k, v in c.items() if k in valid_keys}
                        cam = CameraInfo(**filtered)
                        self.cameras[cam.id] = cam
                logger.info(f"Loaded {len(self.cameras)} cameras from {self.config_path}")
                return
            except Exception as e:
                logger.error(f"Error reading {self.config_path}: {e}")

        # Fallback defaults: real physical USB camera and real CCTV IP stream
        default_cams = [
            CameraInfo(id="usb_0", name="Local Safety Webcam", type="usb", source="0", location="Safety Desk"),
            CameraInfo(
                id="cctv_entrance",
                name="Dedicated IP CCTV (RTSP)",
                type="ip",
                source="rtsp://admin:pass@192.168.1.100:554/live",
                location="Main Entrance",
                enabled=True,
            ),
        ]
        for c in default_cams:
            self.cameras[c.id] = c
        self.save_config()

    def save_config(self):
        """Save current cameras and active selection to cameras.json."""
        try:
            payload = {
                "active_camera_id": self.active_camera_id,
                "cameras": [c.to_dict() for c in self.cameras.values()],
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"Saved camera configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save {self.config_path}: {e}")

    def scan_cameras(self) -> List[Dict[str, Any]]:
        """Quickly probe all configured cameras and discover local USB devices."""
        logger.info("Initiating dynamic camera scan...")
        results = []

        # 1. Probe all cameras in registry
        for cam_id, cam in self.cameras.items():
            alive, msg, res = probe_source(cam.source)
            cam.status = "CONNECTED" if alive else "NOT_CONNECTED"
            cam.error_message = msg if not alive else ""
            if alive:
                cam.resolution = res
                cam.last_seen = time.time()
            results.append(cam.to_dict())

        # 2. Discover any additional USB webcams not yet in registry
        for idx in range(4):
            idx_str = str(idx)
            existing = any(c.source == idx_str for c in self.cameras.values())
            if not existing:
                alive, msg, res = probe_source(idx_str, timeout=1.0)
                if alive:
                    new_id = f"usb_{idx}"
                    new_cam = CameraInfo(
                        id=new_id,
                        name=f"Discovered USB Webcam {idx}",
                        type="usb",
                        source=idx_str,
                        location="Local Port",
                        status="CONNECTED",
                        resolution=res,
                        last_seen=time.time(),
                    )
                    self.cameras[new_id] = new_cam
                    results.append(new_cam.to_dict())

        self.save_config()
        return results

    def get_active_stream(self, custom_source: Optional[str] = None) -> ResilientCameraStream:
        """Get or initialize the resilient stream for the active camera."""
        if custom_source:
            # Check if this custom source matches an existing camera or create ad-hoc
            match = next((c for c in self.cameras.values() if c.source == custom_source), None)
            if not match:
                adhoc_id = f"custom_{int(time.time())}"
                cam_type = "ip" if "://" in custom_source else ("usb" if custom_source.isdigit() else "file")
                match = CameraInfo(
                    id=adhoc_id,
                    name=f"Custom Stream ({custom_source})",
                    type=cam_type,
                    source=custom_source,
                    location="Dynamic CLI",
                )
                self.cameras[adhoc_id] = match
            self.active_camera_id = match.id

        target_cam = self.cameras.get(self.active_camera_id)
        if not target_cam:
            # Fallback to first available
            target_cam = next(iter(self.cameras.values()))
            self.active_camera_id = target_cam.id

        if self.active_stream is not None:
            if self.active_stream.config.id == target_cam.id and self.active_stream.running:
                return self.active_stream
            self.active_stream.stop()

        self.active_stream = ResilientCameraStream(target_cam)
        self.active_stream.start()
        return self.active_stream

    def switch_camera(self, camera_id: str) -> bool:
        """Switch active camera feed dynamically."""
        if camera_id not in self.cameras:
            logger.error(f"Cannot switch to unknown camera '{camera_id}'")
            return False

        logger.info(f"Switching active camera to '{camera_id}' ({self.cameras[camera_id].name})")
        self.active_camera_id = camera_id
        if self.active_stream is not None:
            self.active_stream.stop()
        self.active_stream = ResilientCameraStream(self.cameras[camera_id])
        self.active_stream.start()
        self.save_config()
        return True

    def cycle_next_camera(self) -> str:
        """Cycle to next configured camera (useful for keyboard hotkeys)."""
        keys = list(self.cameras.keys())
        if not keys:
            return ""
        try:
            curr_idx = keys.index(self.active_camera_id)
            next_idx = (curr_idx + 1) % len(keys)
        except ValueError:
            next_idx = 0
        self.switch_camera(keys[next_idx])
        return keys[next_idx]

    def get_status_summary(self) -> Dict[str, Any]:
        """Generate full JSON status report for dashboard and API."""
        return {
            "active_camera_id": self.active_camera_id,
            "total_cameras": len(self.cameras),
            "connected_cameras": sum(1 for c in self.cameras.values() if c.status == "CONNECTED"),
            "cameras": [c.to_dict() for c in self.cameras.values()],
        }


if __name__ == "__main__":
    print("Testing CameraManager dynamic discovery...")
    mgr = CameraManager()
    summary = mgr.scan_cameras()
    print(json.dumps(mgr.get_status_summary(), indent=2))
