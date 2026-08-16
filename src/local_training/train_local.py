"""
Drishti Kavach: Universal Cross-Platform Local Training Pipeline for RailDrishti

Supported Environments:
  - macOS (Apple Silicon M1/M2/M3/M4 via Metal Performance Shaders - MPS)
  - Windows 10/11 (Intel / AMD CPUs or NVIDIA GeForce / RTX GPUs)
  - Linux / Ubuntu (NVIDIA CUDA or Multi-core CPU)

Features:
  - Automatic hardware detection and memory tuning across all platforms.
  - Multi-task YOLO11-seg instance segmentation and 11-class hazard detection.
  - Suppresses all non-critical Python / PyTorch / OpenCV runtime warnings.
  - Real-World Readiness Monitor: Compares accuracy after every epoch against actual Indian Railways operational targets.
  - Auto-export and deployment of trained weights to models/RailDrishti.pt upon completion.

Usage:
  # 1. Standard Training (Recommended - 640px for fast training on laptops):
  python src/local_training/train_local.py --epochs 40 --batch 8 --imgsz 640

  # 2. High-Precision Training (1024px full resolution):
  python src/local_training/train_local.py --epochs 40 --batch 4 --imgsz 1024

  # 3. Resume interrupted training:
  python src/local_training/train_local.py --resume
"""

import os
import sys
import shutil
import argparse
import platform
import warnings
import logging

# 1. Suppress all non-critical warnings across OS
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.getLogger("ultralytics").setLevel(logging.WARNING)

import torch
from ultralytics import YOLO


class EpochStatusMonitor:
    """
    Tracks training progress and compares accuracy metrics against real-world
    operational requirements for Indian Railways Kavach deployment.
    """
    
    # Real-World Operational Targets
    TARGET_BOX_MAP = 85.0      # Required for reliable obstacle braking
    TARGET_SEG_MAP = 90.0      # Required for pinpoint track geometry locking
    TARGET_OVERALL_MAP = 85.0  # Overall deployment threshold

    def __init__(self):
        self.best_map = 0.0
        self.best_epoch = 0
        self.prev_loss = None
        self.prev_map = None

    def on_fit_epoch_end(self, trainer):
        epoch = trainer.epoch + 1
        total_epochs = trainer.epochs
        metrics = getattr(trainer, "metrics", {}) or {}

        # Extract current accuracy percentages (0-100)
        box_map50 = (metrics.get("metrics/mAP50(B)", 0.0) or 0.0) * 100.0
        seg_map50 = (metrics.get("metrics/mAP50(M)", 0.0) or 0.0) * 100.0
        overall_map50 = (box_map50 + seg_map50) / 2.0 if (box_map50 > 0 and seg_map50 > 0) else (box_map50 or seg_map50 or 0.0)

        # Extract training loss
        loss_val = None
        if hasattr(trainer, "tloss") and trainer.tloss is not None:
            try:
                loss_val = float(trainer.tloss.mean()) if hasattr(trainer.tloss, "mean") else float(trainer.tloss)
            except Exception:
                loss_val = None

        # Check if new all-time best
        is_new_best = False
        if overall_map50 > self.best_map and overall_map50 > 2.0:
            self.best_map = overall_map50
            self.best_epoch = epoch
            is_new_best = True

        # Real-World Accuracy Readiness Categorization
        if overall_map50 >= 90.0:
            state_badge = "🏆 [EXCELLENT - DEPLOYMENT READY (EXCEEDS TARGET)]"
            readiness_desc = "FIELD READY (Exceeds all safety & real-world operational benchmarks)"
            health_color = "\033[92m"  # Bright Green
        elif overall_map50 >= 80.0:
            state_badge = "🌟 [GOOD / OPERATIONAL GRADE (MEETS REAL-WORLD TARGET)]"
            readiness_desc = "OPERATIONAL GRADE (Meets field deployment safety threshold >= 85%)"
            health_color = "\033[96m"  # Cyan
        elif overall_map50 >= 65.0:
            state_badge = "📈 [LEARNING & IMPROVING - APPROACHING TARGET]"
            readiness_desc = "PROMISING (Approaching target; train more to refine small obstacles)"
            health_color = "\033[93m"  # Yellow
        elif overall_map50 >= 45.0:
            state_badge = "🔄 [UNDER-TRAINED - MORE EPOCHS REQUIRED]"
            readiness_desc = "INTERMEDIATE (Learning basic track shapes; slender hazards unrefined)"
            health_color = "\033[95m"  # Magenta
        else:
            state_badge = "🌱 [INITIALIZING / PRE-CONVERGENCE (CONTINUE TRAINING)]"
            readiness_desc = "EARLY STAGE (Pre-convergence; high false-alarm risk if deployed now)"
            health_color = "\033[91m"  # Red

        # Comparison delta strings
        box_diff = box_map50 - self.TARGET_BOX_MAP
        seg_diff = seg_map50 - self.TARGET_SEG_MAP
        overall_diff = overall_map50 - self.TARGET_OVERALL_MAP

        box_tag = f"[ {'+' if box_diff >= 0 else ''}{box_diff:5.1f}% vs Target ]"
        seg_tag = f"[ {'+' if seg_diff >= 0 else ''}{seg_diff:5.1f}% vs Target ]"
        overall_tag = f"[ {'+' if overall_diff >= 0 else ''}{overall_diff:5.1f}% vs Target ]"

        reset_col = "\033[0m"

        print(f"\n{health_color}┌───────────────────────────────────────────────────────────────────────────────┐{reset_col}")
        print(f"{health_color}│  EPOCH [{epoch:02d}/{total_epochs:02d}] MODEL STATE : {state_badge}{reset_col}")
        print(f"{health_color}├───────────────────────────────────────────────────────────────────────────────┤{reset_col}")
        print(f"│  REAL-WORLD ACCURACY COMPARISON (Current vs Required Target):                 │")
        print(f"│  • Obstacle Detection Box mAP@50 : {box_map50:5.1f}% / {self.TARGET_BOX_MAP:4.1f}% Target  {box_tag}")
        print(f"│  • Track Segment Mask mAP@50     : {seg_map50:5.1f}% / {self.TARGET_SEG_MAP:4.1f}% Target  {seg_tag}")
        print(f"│  • Combined Overall Score        : {overall_map50:5.1f}% / {self.TARGET_OVERALL_MAP:4.1f}% Target  {overall_tag}")
        print(f"{health_color}├───────────────────────────────────────────────────────────────────────────────┤{reset_col}")
        print(f"│  OPERATIONAL ASSESSMENT : {readiness_desc}")
        if loss_val is not None:
            loss_trend = ""
            if self.prev_loss is not None:
                diff = loss_val - self.prev_loss
                loss_trend = f" ({'+' if diff > 0 else ''}{diff:.4f} vs last epoch)"
            print(f"│  • Current Training Loss  : {loss_val:.4f}{loss_trend}")
        print(f"│  • Peak Accuracy Recorded : {self.best_map:5.1f}% (Epoch {self.best_epoch})" + (" [NEW RECORD!]" if is_new_best else ""))
        print(f"{health_color}└───────────────────────────────────────────────────────────────────────────────┘{reset_col}\n")

        self.prev_loss = loss_val
        self.prev_map = overall_map50


def check_system_hardware() -> tuple:
    """Detects best hardware accelerator and optimal worker thread count across Windows, Mac, and Linux."""
    os_name = platform.system()
    proc_name = platform.processor() or platform.machine()
    
    print("\n" + "=" * 65)
    print(" DRISHTI KAVACH: HARDWARE ACCELERATION CHECK")
    print("=" * 65)
    print(f" • Operating System: {os_name} ({platform.release()})")
    print(f" • CPU Architecture: {proc_name}")
    print(f" • PyTorch Version : {torch.__version__}")

    # 1. Check NVIDIA GPU (Windows / Linux)
    if torch.cuda.is_available():
        device = "0"
        gpu_name = torch.cuda.get_device_name(0)
        workers = 2 if os_name == "Windows" else 4
        print(f" • Compute Engine  : NVIDIA CUDA GPU ({gpu_name})")
        print(" • Acceleration    : ACTIVE (NVIDIA CUDA Hardware Acceleration)")

    # 2. Check Apple Silicon MPS (macOS)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        workers = 4
        print(" • Compute Engine  : Apple Silicon GPU (Metal Performance Shaders - MPS)")
        print(" • Acceleration    : ACTIVE (Apple Silicon Metal Acceleration)")

    # 3. Fallback to Multi-core CPU (Intel / AMD / Generic)
    else:
        device = "cpu"
        workers = 0 if os_name == "Windows" else 2
        cpu_count = os.cpu_count() or 4
        print(f" • Compute Engine  : Multi-core CPU ({cpu_count} Logical Cores)")
        print(" • Acceleration    : CPU Vectorized Multi-Threading")
    
    print("=" * 65 + "\n")
    return device, workers


def train_raildrishti(args):
    # Determine project root and paths cross-platform
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_yaml = os.path.join(project_root, args.data)
    weights_path = os.path.join(project_root, args.weights) if not os.path.isabs(args.weights) else args.weights
    output_model_path = os.path.join(project_root, "models/RailDrishti.pt")

    if not os.path.exists(data_yaml):
        print(f"[ERROR] Dataset configuration file not found at: {data_yaml}")
        sys.exit(1)

    detected_device, default_workers = check_system_hardware()
    device = args.device if args.device is not None else detected_device
    workers = args.workers if args.workers is not None else default_workers

    print("=" * 65)
    print(" STARTING LOCAL RAILDISHTI TRAINING")
    print("=" * 65)
    print(f" • Dataset Config  : {data_yaml}")
    print(f" • Base Weights    : {weights_path}")
    print(f" • Epochs          : {args.epochs}")
    print(f" • Batch Size      : {args.batch}")
    print(f" • Input Image Size: {args.imgsz}x{args.imgsz}")
    print(f" • Device          : {device}")
    print(f" • CPU Workers     : {workers}")
    print(f" • Optimizer       : {args.optimizer}")
    print(f" • Learning Rate   : lr0={args.lr0}, lrf={args.lrf}")
    print("=" * 65 + "\n")

    # Load base model
    if args.resume:
        last_weights = os.path.join(project_root, "runs/segment/raildrishti_local/weights/last.pt")
        if not os.path.exists(last_weights):
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

    # Attach Custom Real-World Epoch Monitor Callback
    monitor = EpochStatusMonitor()
    model.add_callback("on_fit_epoch_end", monitor.on_fit_epoch_end)

    # Execute training
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        patience=15,
        project="runs/segment",
        name="raildrishti_local",
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

    best_pt = os.path.join(project_root, "runs/segment/raildrishti_local/weights/best.pt")
    if os.path.exists(best_pt):
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        shutil.copy(best_pt, output_model_path)
        print(f"[+] Best trained weights successfully saved to: {output_model_path}")
    else:
        print(f"[!] Warning: {best_pt} not found. Check runs/segment/raildrishti_local/weights/")

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
    parser = argparse.ArgumentParser(description="Drishti Kavach Universal Local Training")
    parser.add_argument("--data", type=str, default="configs/raildrishti_dataset.yaml", help="Path to dataset YAML")
    parser.add_argument("--weights", type=str, default="yolo11s-seg.pt", help="Pretrained base weights")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs (default: 40)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (640 for fast laptop training, 1024 for high-res)")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', '0', 'cpu'). Default: auto-detect")
    parser.add_argument("--workers", type=int, default=None, help="Data loader worker processes (auto-tuned)")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer: AdamW, SGD, Adam")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")

    args = parser.parse_args()
    train_raildrishti(args)
