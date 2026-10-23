"""
Convert HEIC → JPEG and extract video frames into a training-ready folder.

Output structure:
  Data/labeled_ready/
    person_001.jpg, person_002.jpg, ...
    hat_001.jpg, hat_002.jpg, ...
    vest_001.jpg, vest_002.jpg, ...
    hatvest_001.jpg, hatvest_002.jpg, ...
    items_001.jpg, items_002.jpg, ...
    video_frame_001.jpg, video_frame_002.jpg, ...
"""

import sys
from pathlib import Path

try:
    from PIL import Image
    from pillow_heif import register_heif_opener
    register_heif_opener()          # lets PIL open .heic/.heif natively
except ImportError:
    print("ERROR: pip install pillow-heif Pillow")
    sys.exit(1)

import cv2

DATA_DIR = Path(__file__).parent / "Data"
OUT_DIR  = DATA_DIR / "labeled_ready"
OUT_DIR.mkdir(exist_ok=True)

JPEG_QUALITY = 95   # high quality for labeling

# ── Mapping: folder → output prefix ─────────────────────────────────
FOLDER_MAP = {
    "Person":      "person",
    "WithHat":     "hat",
    "WithVest":    "vest",
    "WithHatVest": "hatvest",
    "Items":       "items",
}

converted = 0

# ── 1. Convert images (HEIC + existing JPEG) ────────────────────────
for folder_name, prefix in FOLDER_MAP.items():
    src_dir = DATA_DIR / folder_name
    if not src_dir.exists():
        print(f"  ⚠  Skipping {folder_name}/ (not found)")
        continue

    files = sorted(src_dir.iterdir())
    idx = 1
    for f in files:
        ext = f.suffix.lower()
        if ext not in (".heic", ".heif", ".jpg", ".jpeg", ".png"):
            continue
        try:
            img = Image.open(f)
            img = img.convert("RGB")        # strip alpha / ICC issues

            # Resize to max 1280px longest side (YOLO training std)
            max_side = max(img.size)
            if max_side > 1280:
                scale = 1280 / max_side
                new_w = int(img.size[0] * scale)
                new_h = int(img.size[1] * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)

            out_path = OUT_DIR / f"{prefix}_{idx:03d}.jpg"
            img.save(out_path, "JPEG", quality=JPEG_QUALITY)
            idx += 1
            converted += 1
        except Exception as e:
            print(f"  ✗  Failed {f.name}: {e}")

    print(f"  ✓  {folder_name}/ → {idx - 1} images  (prefix: {prefix}_)")


# ── 2. Extract video frames ─────────────────────────────────────────
video_dir = DATA_DIR / "Video"
frame_count = 0

if video_dir.exists():
    for vf in sorted(video_dir.glob("*.mp4")):
        cap = cv2.VideoCapture(str(vf))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample every 0.5 seconds (not every frame — avoids near-duplicates)
        sample_interval = max(1, int(fps * 0.5))

        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx % sample_interval == 0:
                # Resize to max 1280
                h, w = frame.shape[:2]
                max_side = max(h, w)
                if max_side > 1280:
                    scale = 1280 / max_side
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                frame_count += 1
                out_path = OUT_DIR / f"video_frame_{frame_count:03d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            f_idx += 1
        cap.release()
        print(f"  ✓  {vf.name} → {frame_count} frames  (sampled every {sample_interval} frames)")

# ── Summary ──────────────────────────────────────────────────────────
total = converted + frame_count
print(f"\n{'='*50}")
print(f"  Total images ready for labeling: {total}")
print(f"  Output: {OUT_DIR}")
print(f"{'='*50}")
