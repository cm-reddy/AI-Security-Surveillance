# AI Security & Safety Surveillance System

A computer vision system for real-time Personal Protective Equipment (PPE) detection, restricted zone monitoring, and CCTV stream supervision. Built on a self-trained YOLOv8 neural network and an OpenCV video processing pipeline.

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
                      │  - Far-distance CCTV crop zoom analysis      │
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
                      │  - Dynamic camera health grid (FPS/status)   │
                      │  - CSV incident audit logging (events.csv)   │
                      └──────────────────────────────────────────────┘
```

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

---

## Camera Configuration (`cameras.json`)

All camera endpoints are defined in [`cameras.json`](file:///c:/Users/sachi/Downloads/ppe_webcam_poc/cameras.json).

```json
{
  "active_camera_id": "usb_0",
  "cameras": [
    {
      "id": "usb_0",
      "name": "Local Safety Webcam",
      "type": "usb",
      "source": "0",
      "location": "Control Room / Safety Desk",
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
python ppe_webcam_poc.py --camera 1   # External USB webcam
```
*(Note: Ensure you use the singular flag `--camera`, not `--cameras`).*

### 5. Launch the Web Command Center Dashboard
In a separate terminal, start the local monitoring dashboard:
```bash
python dashboard.py --port 8080
```
Open **[http://localhost:8080](http://localhost:8080)** in any browser.

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

---

## Stream Fault Tolerance & Connection Handling

- **Fast TCP Pre-Check:** Network endpoints are tested with a 0.8s socket handshake before opening video capture, preventing OpenCV thread locking.
- **Buffer Flush:** Uses `nobuffer` and low packet buffering to prevent latency accumulation on continuous RTSP CCTV streams.
- **Disconnected Fallback Screen:** If a network drop occurs, the OpenCV display switches to an offline diagnostic frame with a countdown timer, attempting background reconnection without terminating the application.
- **Far-Distance CCTV Zoom:** The `--enhance-distant-workers` flag crops distant worker regions (height 45–300px) and applies secondary high-resolution inference to ensure helmet and vest detection at long ranges.

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
