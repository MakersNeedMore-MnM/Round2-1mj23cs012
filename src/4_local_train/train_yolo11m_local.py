"""
Drishti-Kavach: Local YOLO11m Obstacle Detection Trainer (Apple Silicon MPS / CUDA / CPU)

Usage:
  # Quick test run:
  python src/4_local_train/train_yolo11m_local.py --epochs 3 --batch-size 4 --imgsz 640

  # Standard training:
  python src/4_local_train/train_yolo11m_local.py --epochs 30 --batch-size 8 --imgsz 1024
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def train_yolo11m_local(
    data_yaml: str = "dataset_detection/detection_data.yaml",
    save_dir: str = "models",
    epochs: int = 30,
    batch_size: int = 8,
    imgsz: int = 1024,
    device_choice: str = "auto"
):
    if device_choice == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "0"
        else:
            device = "cpu"
    else:
        device = device_choice

    out_models_dir = Path(save_dir)
    out_models_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        print(f"[!] Error: {yaml_path} not found. Run prep_obstacle_detection.py first.")
        return

    print("=" * 75)
    print(" 🛡️ DRISHTI-KAVACH: LOCAL YOLO11m OBSTACLE DETECTOR TRAINING")
    print("=" * 75)
    print(f" • Device Selected:    {device}")
    print(f" • Dataset YAML:       {yaml_path}")
    print(f" • Target Resolution:  {imgsz}x{imgsz}")
    print(f" • Batch Size:         {batch_size}")
    print(f" • Total Epochs:       {epochs}")
    print("=" * 75)

    base_weights = "models/yolo11m.pt" if Path("models/yolo11m.pt").exists() else "yolo11m.pt"
    model = YOLO(base_weights)

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        save=True,
        project="runs",
        name="yolo11m_local",
        exist_ok=True
    )

    best_pt = Path("runs/yolo11m_local/weights/best.pt")
    if best_pt.exists():
        target_pt = out_models_dir / "best_yolo11m_local.pt"
        import shutil
        shutil.copy(str(best_pt), str(target_pt))
        print("=" * 75)
        print(f" 🎉 LOCAL YOLO11m TRAINING COMPLETE!")
        print(f" • Best Weights: {target_pt}")
        print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local YOLO11m Trainer for Railway Obstacles")
    parser.add_argument("--data", type=str, default="dataset_detection/detection_data.yaml", help="Path to detection_data.yaml")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save model weights")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/mps/0/cpu)")

    args = parser.parse_args()
    train_yolo11m_local(
        data_yaml=args.data,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device_choice=args.device
    )
