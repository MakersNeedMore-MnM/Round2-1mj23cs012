"""
Drishti Kavach: Local Mac / Apple Silicon Training Pipeline for RailDrishti11-Seg

Features:
  - Apple Silicon GPU (Metal Performance Shaders - MPS) hardware acceleration.
  - Multi-task YOLO11-seg instance segmentation and 11-class hazard detection.
  - Suppresses all non-critical Python / PyTorch / OpenCV runtime warnings.
  - Custom Epoch Health Monitor: Real-time assessment after every epoch (BEST, LEARNING, CONVERGING).
  - Auto-export and deployment of trained weights to models/RailDrishti.pt upon completion.

Usage:
  # 1. Standard Training (Recommended on Mac - 640px for fast training):
  python src/local_training/train_local_mac.py --epochs 40 --batch 8 --imgsz 640

  # 2. High-Precision Training (1024px full resolution):
  python src/local_training/train_local_mac.py --epochs 40 --batch 4 --imgsz 1024

  # 3. Resume interrupted training:
  python src/local_training/train_local_mac.py --resume
"""

import os
import sys
import shutil
import argparse
import warnings
import logging

# 1. Suppress all non-critical warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.getLogger("ultralytics").setLevel(logging.WARNING)

import torch
from ultralytics import YOLO


class EpochStatusMonitor:
    """Tracks training progress and prints qualitative health/learning status after every epoch."""
    
    def __init__(self):
        self.best_map = 0.0
        self.prev_loss = None
        self.prev_map = None

    def on_fit_epoch_end(self, trainer):
        epoch = trainer.epoch + 1
        total_epochs = trainer.epochs
        metrics = getattr(trainer, "metrics", {}) or {}

        # Extract metrics
        box_map50 = metrics.get("metrics/mAP50(B)", 0.0) or 0.0
        seg_map50 = metrics.get("metrics/mAP50(M)", 0.0) or 0.0
        overall_map50 = (box_map50 + seg_map50) / 2.0 if (box_map50 and seg_map50) else (box_map50 or seg_map50 or 0.0)

        # Extract loss if available
        loss_val = None
        if hasattr(trainer, "tloss") and trainer.tloss is not None:
            try:
                loss_val = float(trainer.tloss.mean()) if hasattr(trainer.tloss, "mean") else float(trainer.tloss)
            except Exception:
                loss_val = None

        # Determine Model Learning State
        is_best = False
        if overall_map50 > self.best_map and overall_map50 > 0.02:
            self.best_map = overall_map50
            is_best = True

        if is_best:
            state_badge = "🌟 [BEST MODEL SO FAR - PEAK ACCURACY]"
            health_color = "\033[92m"  # Bright Green
        elif self.prev_map is not None and overall_map50 > self.prev_map:
            state_badge = "📈 [LEARNING & IMPROVING - ACCURACY UP]"
            health_color = "\033[96m"  # Cyan
        elif self.prev_loss is not None and loss_val is not None and loss_val < self.prev_loss:
            state_badge = "🔄 [OPTIMIZING WEIGHTS - LOSS DECREASING]"
            health_color = "\033[94m"  # Blue
        elif epoch <= 3:
            state_badge = "🌱 [WARMUP & FEATURE INITIALIZATION]"
            health_color = "\033[93m"  # Yellow
        else:
            state_badge = "⚡ [STEADY / CONVERGING STATE]"
            health_color = "\033[97m"  # White

        reset_col = "\033[0m"

        print(f"\n{health_color}┌────────────────────────────────────────────────────────────────────────┐{reset_col}")
        print(f"{health_color}│  EPOCH [{epoch:02d}/{total_epochs:02d}] MODEL STATE : {state_badge}{reset_col}")
        print(f"{health_color}├────────────────────────────────────────────────────────────────────────┤{reset_col}")
        print(f"│  • Obstacle Detection Box mAP@50 : {box_map50 * 100:5.1f}%")
        print(f"│  • Track Segment Mask mAP@50     : {seg_map50 * 100:5.1f}%")
        print(f"│  • Combined Mean mAP@50          : {overall_map50 * 100:5.1f}% (All-Time Best: {self.best_map * 100:5.1f}%)")
        if loss_val is not None:
            loss_trend = ""
            if self.prev_loss is not None:
                diff = loss_val - self.prev_loss
                loss_trend = f" ({'+' if diff > 0 else ''}{diff:.4f})"
            print(f"│  • Current Epoch Loss            : {loss_val:.4f}{loss_trend}")
        print(f"{health_color}└────────────────────────────────────────────────────────────────────────┘{reset_col}\n")

        self.prev_loss = loss_val
        self.prev_map = overall_map50


def check_system_hardware() -> str:
    """Detects best hardware accelerator on macOS (MPS, CUDA, or CPU)."""
    print("\n" + "=" * 65)
    print(" DRISHTI KAVACH: HARDWARE ACCELERATION CHECK")
    print("=" * 65)
    print(f" • PyTorch Version : {torch.__version__}")

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        print(" • Compute Engine  : Apple Silicon GPU via Metal Performance Shaders (MPS)")
        print(" • Acceleration    : ACTIVE (Hardware Accelerated)")
    elif torch.cuda.is_available():
        device = "0"
        print(f" • Compute Engine  : NVIDIA CUDA GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print(" • Compute Engine  : CPU (Multi-core Fallback)")
    
    print("=" * 65 + "\n")
    return device


def train_raildrishti(args):
    # Determine project root and paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_yaml = os.path.join(project_root, args.data)
    weights_path = os.path.join(project_root, args.weights) if not os.path.isabs(args.weights) else args.weights
    output_model_path = os.path.join(project_root, "models/RailDrishti.pt")

    if not os.path.exists(data_yaml):
        print(f"[ERROR] Dataset configuration file not found at: {data_yaml}")
        sys.exit(1)

    device = args.device if args.device is not None else check_system_hardware()

    print("=" * 65)
    print(" STARTING LOCAL RAILDISHTI11-SEG TRAINING")
    print("=" * 65)
    print(f" • Dataset Config  : {data_yaml}")
    print(f" • Base Weights    : {weights_path}")
    print(f" • Epochs          : {args.epochs}")
    print(f" • Batch Size      : {args.batch}")
    print(f" • Input Image Size: {args.imgsz}x{args.imgsz}")
    print(f" • Device          : {device}")
    print(f" • CPU Workers     : {args.workers}")
    print(f" • Optimizer       : {args.optimizer}")
    print(f" • Learning Rate   : lr0={args.lr0}, lrf={args.lrf}")
    print("=" * 65 + "\n")

    # Load base model
    if args.resume:
        last_weights = os.path.join(project_root, "runs/segment/raildrishti_mac/weights/last.pt")
        if not os.path.exists(last_weights):
            print(f"[ERROR] Cannot resume: checkpoint not found at {last_weights}")
            sys.exit(1)
        print(f"[+] Resuming training from checkpoint: {last_weights}")
        model = YOLO(last_weights)
        resume_flag = True
    else:
        model = YOLO(weights_path)
        resume_flag = False

    # Attach Custom Epoch Monitor Callback
    monitor = EpochStatusMonitor()
    model.add_callback("on_fit_epoch_end", monitor.on_fit_epoch_end)

    # Execute training
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        patience=15,
        project="runs/segment",
        name="raildrishti_mac",
        exist_ok=True,
        save=True,
        save_period=5,
        plots=True,
        verbose=True,
        resume=resume_flag
    )

    print("\n" + "=" * 65)
    print(" TRAINING COMPLETE: EXPORTING WEIGHTS")
    print("=" * 65)

    best_pt = os.path.join(project_root, "runs/segment/raildrishti_mac/weights/best.pt")
    if os.path.exists(best_pt):
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        shutil.copy(best_pt, output_model_path)
        print(f"[+] Best trained weights successfully saved to: {output_model_path}")
    else:
        print(f"[!] Warning: {best_pt} not found. Check runs/segment/raildrishti_mac/weights/")

    # Run quick validation on validation split
    print("\nEvaluating trained model on validation set...")
    val_metrics = model.val(data=data_yaml, imgsz=args.imgsz, device=device)

    print("\n" + "=" * 65)
    print(" FINAL QUANTITATIVE VALIDATION SUMMARY")
    print("=" * 65)
    try:
        box_map50 = val_metrics.box.map50
        box_map = val_metrics.box.map
        seg_map50 = val_metrics.seg.map50
        seg_map = val_metrics.seg.map
        print(f" • Obstacle Bounding Box mAP@50     : {box_map50 * 100:5.2f}%")
        print(f" • Obstacle Bounding Box mAP@50-95  : {box_map * 100:5.2f}%")
        print(f" • Track Mask Segmentation mAP@50   : {seg_map50 * 100:5.2f}%")
        print(f" • Track Mask Segmentation mAP@50-95 : {seg_map * 100:5.2f}%")
    except Exception as e:
        print(f" • Metrics Summary: {val_metrics}")

    print("=" * 65)
    print(f"\n[+] You can now run live inference using:")
    print(f"    python run_inference.py --source 0 --model models/RailDrishti.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drishti Kavach Local Mac Training")
    parser.add_argument("--data", type=str, default="configs/raildrishti_dataset.yaml", help="Path to dataset YAML")
    parser.add_argument("--weights", type=str, default="yolo11s-seg.pt", help="Pretrained base weights")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs (default: 40)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8 for Mac)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (640 for fast Mac training, 1024 for high-res)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('mps', 'cpu', '0'). Default: auto-detect")
    parser.add_argument("--workers", type=int, default=4, help="Data loader CPU worker processes")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer: AdamW, SGD, Adam")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")

    args = parser.parse_args()
    train_raildrishti(args)
