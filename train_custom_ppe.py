#!/usr/bin/env python3
"""Fine-tune YOLOv8 on custom PPE dataset and prepare weights for the POC pipeline.

Usage:
    python train_custom_ppe.py --data path/to/data.yaml --epochs 50 --batch 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train custom PPE model for webcam POC")
    parser.add_argument("--data", required=True, help="Path to data.yaml (from Roboflow or CVAT export)")
    parser.add_argument("--base-model", default="yolov8s.pt", help="Pretrained base weights (default: yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16 on GPU; use 8 or 4 for low VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu", help="Device to train on (e.g. 0, cpu)")
    parser.add_argument("--output-dir", default="weights", help="Directory to save final model weights")
    return parser.parse_args()


def main():
    args = parse_args()
    data_yaml = Path(args.data).resolve()

    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found at {data_yaml}", file=sys.stderr)
        print("Please export your labeled dataset from Roboflow in 'YOLOv8' format and point --data to data.yaml.")
        return 1

    device_str = args.device
    if device_str in ("0", 0) and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device_str = f"GPU:0 ({gpu_name})"
    else:
        device_str = f"CPU / {args.device}"

    print("=" * 65)
    print(" CUSTOM PPE MODEL TRAINING PIPELINE")
    print("=" * 65)
    print(f"Dataset config : {data_yaml}")
    print(f"Base weights   : {args.base_model}")
    print(f"Epochs         : {args.epochs}")
    print(f"Image size     : {args.imgsz}")
    print(f"Batch size     : {args.batch}")
    print(f"Target device  : {device_str}")
    print("=" * 65)

    # 1. Load pretrained base model for transfer learning
    print("\n[1/3] Initializing base model...")
    model = YOLO(args.base_model)

    # 2. Train on custom dataset
    print("\n[2/3] Training on your custom images...")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=2,           # Keep workers low on Windows to prevent RAM spikes
        patience=15,          # Early stopping if accuracy plateaus
        save=True,
        project="runs/detect",
        name="custom_ppe",
        exist_ok=True,
        verbose=True,
    )

    # 3. Locate best.pt and copy to weights directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_weights = Path(results.save_dir) / "weights" / "best.pt"

    dest_weights = out_dir / "my_custom_ppe.pt"
    if best_weights.exists():
        import shutil
        shutil.copy2(best_weights, dest_weights)
        print("\n" + "=" * 65)
        print(" TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 65)
        print(f"Trained model saved to: {dest_weights.resolve()}")
        print("\nTo launch the monitoring POC with your new model, run:")
        print(f"  python ppe_webcam_poc.py --model {dest_weights}")
        print("=" * 65)
        return 0
    else:
        print(f"WARNING: Training completed, but best.pt was not found at {best_weights}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
