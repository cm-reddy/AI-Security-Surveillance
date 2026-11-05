# AI Security & Safety Surveillance System

A computer vision system for real-time Personal Protective Equipment (PPE) detection, restricted zone monitoring, and CCTV stream supervision. Built on a self-trained YOLOv8 neural network, an OpenCV video processing pipeline, and a **Zero-Trust safety compliance model**.

Supports dedicated IP cameras (RTSP/HTTP), USB video devices, dynamic hardware discovery, and real-time dashboard monitoring.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │            Video Ingestion Layer             │
                      │  - Dedicated IP CCTV (RTSP / HTTP / ONVIF)   │
                      │  - USB Webcams (Dynamic Port Discovery)      │
                      │  - Non-blocking TCP pre-ping & health checks │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │              AI Inference Engine             │
                      │  - Self-trained YOLOv8 (weights/my_custom_ppe.pt)
                      │  - 6 Classes: Person, Helmet, No_Helmet,     │
                      │               Vest, No_Vest, BorderLine      │
                      │  - ByteTrack multi-object worker tracking    │
                      │  - Zero-Trust PPE compliance model           │
                      │  - OpenCV crop zoom with INTER_CUBIC upscale │
                      │  - IoU-based spatial track continuity        │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │          Spatial Logic & Safety HUD          │
                      │  - Person-to-PPE bounding box association    │
                      │  - Virtual tripwire / dual-border boundaries │
                      │  - Real-time audible violation alarm         │
                      │  - Failover: "CAMERA NOT CONNECTED" screen   │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │        Command Center & Audit Feeds          │
                      │  - Web Dashboard (http://localhost:8080)     │
                      │  - Dynamic alert banner (green/red)          │
                      │  - CSV incident audit logging (events.csv)   │
                      └──────────────────────────────────────────────┘
```

---

## Zero-Trust Safety Model

The system applies a **Zero-Trust** compliance rule: a worker is classified as **COMPLIANT** only when **both** helmet **and** vest are verified as `PRESENT`. In every other case — missing, unverified, or conflicting evidence — the worker is classified as **NON_COMPLIANT**.

There is no intermediate "UNKNOWN" overall status. This eliminates ambiguity for safety supervisors and ensures any doubt defaults to a safety violation.

### PPE Compliance Cases

| Case | Helmet | Vest | Overall Status | Box Color | Event Type |
|------|--------|------|----------------|-----------|------------|
| Full PPE | PRESENT | PRESENT | **COMPLIANT** | 🟢 Green | `PPE_COMPLIANT_CROSSING` |
| Helmet only | PRESENT | MISSING / UNKNOWN | **NON_COMPLIANT** | 🔴 Red | `PPE_NONCOMPLIANT_CROSSING` |
| Vest only | MISSING / UNKNOWN | PRESENT | **NON_COMPLIANT** | 🔴 Red | `PPE_NONCOMPLIANT_CROSSING` |
| No PPE | MISSING | MISSING | **NON_COMPLIANT** | 🔴 Red | `PPE_NONCOMPLIANT_CROSSING` |

> [!IMPORTANT]
> The model detects **high-visibility safety vests** (class `Vest`), not regular jackets or coats. Workers must wear a proper hi-vis vest to be marked compliant.

---

## System Requirements

- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Python:** Version 3.10 to 3.14
- **Hardware:**
  - Dedicated IP camera (RTSP stream) or standard USB webcam
  - Minimum 4 GB RAM (8 GB recommended)
  - Optional: NVIDIA GPU with CUDA 12.x for accelerated inference (CPU fallback supported)

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ppe_webcam_poc
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Model Weights
Ensure the trained model file exists at `weights/my_custom_ppe.pt`. If you need to retrain, see [Model Training](#model-training-train_custom_ppepy) below.

---

## Camera Configuration (`cameras.json`)

All camera endpoints are defined in [`cameras.json`](file:///c:/Users/sachi/Downloads/ppe_webcam_poc/cameras.json).

```json
{
  "active_camera_id": "usb_0",
  "cameras": [
    {
      "id": "usb_0",
      "name": "Built-in Laptop Webcam",
      "type": "usb",
      "source": "0",
      "location": "Laptop Integrated",
      "enabled": true,
      "reconnect_interval_sec": 3.0
    },
    {
      "id": "usb_2",
      "name": "External USB Webcam",
      "type": "usb",
      "source": "2",
      "location": "USB External Port",
      "enabled": true,
      "reconnect_interval_sec": 3.0
    },
    {
      "id": "cctv_entrance",
      "name": "Dedicated IP CCTV (RTSP)",
      "type": "ip",
      "source": "rtsp://admin:password@192.168.1.100:554/live",
      "location": "Main Entrance / Gate 1",
      "enabled": true,
      "reconnect_interval_sec": 4.0
    }
  ]
}
```

### Field Definitions:
- `id`: Unique identifier for the camera.
- `name`: Display label in UI and logs.
- `type`: `ip` (network stream) or `usb` (direct hardware capture).
- `source`: Device index (`0`, `1`) for USB, or stream URL (`rtsp://...`, `http://...`) for IP cameras.
- `enabled`: Set to `true` to include in active scan cycles.
- `reconnect_interval_sec`: Base retry interval when a connection drops.

---

## Running the System

### 1. Discover Connected Cameras (Dynamic Probe)
Probes local USB ports and configured network IP addresses. Displays reachability status and supported resolutions:
```bash
python ppe_webcam_poc.py --list-cameras
```

### 2. Start Live Monitoring (Default Camera)
Launches the primary stream configured in `cameras.json` (defaults directly to the laptop's built-in webcam `usb_0`):
```bash
python ppe_webcam_poc.py
```

### 3. Connect to a Specific IP Camera (RTSP / HTTP)
Pass any RTSP stream directly via the command line:
```bash
python ppe_webcam_poc.py --camera "rtsp://admin:password@192.168.1.100:554/live"
```

### 4. Connect to a Specific USB Webcam Index
To explicitly target a specific local USB camera index:
```bash
python ppe_webcam_poc.py --camera 0   # Built-in laptop webcam
python ppe_webcam_poc.py --camera 2   # External USB webcam (or cycle with 'C' key live)
```
*(Note: Ensure you use the singular flag `--camera`, not `--cameras`).*

### 5. Recommended Settings for Different Scenarios

| Scenario | Command | Notes |
|----------|---------|-------|
| Close-up webcam (desk) | `python ppe_webcam_poc.py --camera 0` | Default settings work well at short range |
| Wide CCTV / distant workers | `python ppe_webcam_poc.py --camera 0 --imgsz 1280 --conf 0.25` | Higher input resolution + lower confidence for distant workers |
| Debug (show raw detections) | `python ppe_webcam_poc.py --camera 0 --show-detections` | Shows all YOLO bounding boxes before filtering |

### 6. Launch the Web Dashboard
In a **separate terminal**, start the local monitoring dashboard:
```bash
python dashboard.py --port 8080
```
Open **[http://localhost:8080](http://localhost:8080)** in any browser.

### Full Demo (Two Terminals)

**Terminal 1 — Monitoring:**
```bash
python ppe_webcam_poc.py --camera 0
```
**Terminal 2 — Dashboard:**
```bash
python dashboard.py --port 8080
```

> [!TIP]
> Press **D** in the live monitoring window to auto-open the dashboard in your default browser.

---

## Interactive Controls (Live Window)

| Input | Action |
|---|---|
| **C** | Cycle to the next configured camera stream |
| **E** | Toggle in-window event audit drawer |
| **D** | Open web dashboard in default browser |
| **Spacebar** | Trigger manual safety violation test event |
| **[ and ]** | Shift left safety border left / right |
| **, and .** | Shift right safety border left / right |
| **Tab** | Toggle between Dual Vertical Borders and Single Line mode |
| **R** | Reset safety boundaries to default coordinates |
| **Q or Esc** | Terminate monitoring process |
| **Mouse Drag** | Click and drag safety borders directly on video feed |

The HUD shows two clickable buttons at the top of the frame:
- **Events (N)** — Click to toggle the event drawer (where N = event count)
- **Dashboard (D)** — Click to open the web dashboard

---

## Stream Fault Tolerance & Connection Handling

- **Hardware Resolution Auto-Negotiation:** Probes and negotiates the camera's highest native resolution (1080p FHD, 720p HD, or 480p) dynamically, adapting automatically across different laptop hardware.
- **Fast TCP Pre-Check:** Network endpoints are tested with a 0.8s socket handshake before opening video capture, preventing OpenCV thread locking.
- **Buffer Flush:** Uses `nobuffer` and low packet buffering to prevent latency accumulation on continuous RTSP CCTV streams.
- **Disconnected Fallback Screen:** If a network drop occurs, the OpenCV display switches to an offline diagnostic frame with a countdown timer, attempting background reconnection without terminating the application.
- **Far-Distance Crop Enhancement:** When a worker has missing or unverified PPE evidence, their bounding box region is cropped from the full frame, upscaled using OpenCV bicubic interpolation (`cv2.INTER_CUBIC`) to a minimum 320px target, and re-inferred through YOLO at a lower confidence threshold. This recovers helmet and vest detections for workers who appear small on wide-angle CCTV feeds. The crop uses 20% horizontal and 15% vertical padding for better context.
- **Adaptive Confidence Thresholds:** Distant workers (bounding box height ≤ 320px) are accepted at a lower person-detection confidence, while close-up workers use the full confidence threshold. This prevents dropping valid detections at range.
- **IoU-Based Track Continuity:** When ByteTrack reassigns a worker's track ID (due to occlusion or brief frame loss), the system uses spatial IoU overlap (> 0.30) to match the new detection to recent tracks within a 1.8s window, inheriting accumulated PPE evidence history. This prevents compliance status resets from tracker ID churn.

---

## Web Dashboard

The dashboard at `http://localhost:8080` auto-refreshes every 2 seconds and displays:

- **Stats Grid:** Total events, violations, compliant entries, and compliance rate.
- **Dynamic Alert Banner:**
  - 🟢 **Green** — "ALL WORKERS COMPLIANT" when the latest event shows full PPE compliance.
  - 🔴 **Red** — "NON-COMPLIANT" with worker details when a violation is detected.
- **Event Table:** Scrollable list of all logged events with color-coded status tags:
  - **COMPLIANT** (green badge) — Worker verified with both helmet and vest.
  - **NON_COMPLIANT** (red badge) — Any PPE item missing or unverified.
- **Camera Health Grid:** Live FPS and connection status for all configured cameras.

---

## Model Training (`train_custom_ppe.py`)

To fine-tune the YOLOv8 model on custom workplace images:

```bash
python train_custom_ppe.py --data Data/Dataset/data.yaml --epochs 50 --batch 16 --imgsz 640
```

### Custom Model Classes:
- `0: BorderLine` — Demarcation tape / barrier line
- `1: Helmet` — Hardhat present
- `2: No_Helmet` — Unprotected head
- `3: No_Vest` — Unprotected torso
- `4: Person` — Worker detection
- `5: Vest` — High-visibility safety vest present

Trained model weights are automatically saved to `weights/my_custom_ppe.pt`.

---

## Automated Tests

Run the full test suite (safety logic and camera resilience):
```bash
python -m unittest test_ppe.py test_camera_manager.py
```
Expected output:
```text
Ran 24 tests in 1.15s
OK
```

Key test cases:
- **Zero-Trust compliance logic:** Verifies that `overall_status()` returns `COMPLIANT` only when both helmet and vest are `PRESENT`, and `NON_COMPLIANT` for all other combinations.
- **Event logging:** Validates that both compliant and non-compliant events are written to CSV and counted correctly in dashboard stats.
- **Boundary / tripwire logic:** Confirms zone crossing detection and dual-border tracking.
- **Camera manager resilience:** Tests reconnection, health monitoring, and failover screens.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| No detections on distant workers | Default confidence too high for small bounding boxes | Use `--conf 0.25` and `--imgsz 1280` |
| Dashboard shows 0 events | `events.csv` missing or monitoring not running | Start monitoring first, then open dashboard |
| Dashboard shows old "UNKNOWN" events | Legacy events from before Zero-Trust update | Delete or archive `events.csv` to start fresh |
| `?` characters in the HUD | Emoji rendering issue with OpenCV | Already fixed — HUD uses ASCII text labels |
| Track ID keeps changing | ByteTrack losing identity during fast motion | IoU track continuity handles this automatically |
| Worker shown as NON_COMPLIANT despite wearing vest | Model only detects hi-vis safety vests, not regular jackets | Worker must wear a high-visibility safety vest |

---

## Repository Structure

```
ppe_webcam_poc/
├── camera_manager.py            # Stream ingestion, RTSP buffering, and health monitor
├── cameras.json                 # Active camera registry (IP and USB endpoints)
├── convert_data.py              # Frame extraction and HEIC-to-JPEG data converter
├── dashboard.py                 # HTTP server and web monitoring UI
├── events.csv                   # Persistent event audit log
├── ppe_webcam_poc.py            # Main surveillance and PPE detection application
├── requirements.txt             # Python package dependencies
├── test_camera_manager.py       # Camera manager unit tests
├── test_ppe.py                  # Core PPE classification and boundary logic tests
├── train_custom_ppe.py          # YOLOv8 transfer learning script
├── validation_template.csv      # Verification template for accuracy audits
├── Data/                        # Labeled training and validation datasets
├── docs/                        # System documentation assets
└── weights/                     # Pretrained and fine-tuned model weights
```
