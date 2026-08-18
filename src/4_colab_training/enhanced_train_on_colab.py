"""
Drishti Kavach: Enhanced Google Colab Cloud GPU Training & Resumption Pipeline

Features:
  1. Auto Google Drive Mounting & Ultra-Fast NVMe Dataset Extraction.
  2. Persistent Cloud Checkpointing: Saves every epoch directly to Google Drive so progress is NEVER lost.
  3. Interactive & CLI Pause / Resume Mode: Pick up exactly where training left off.
  4. Minimal 3-Line Color-Coded Accuracy Telemetry per epoch.
  5. Permanent Model Export: Automatically saves the best model to 'MyDrive/RailDrishti.pt'.

Usage inside Google Colab:
  # Method 1: Interactive Execution (prompts to Start Fresh or Resume)
  %run src/4_colab_training/enhanced_train_on_colab.py

  # Method 2: Command-Line Execution
  !python src/4_colab_training/enhanced_train_on_colab.py --action resume
  !python src/4_colab_training/enhanced_train_on_colab.py --action fresh --epochs 60 --batch 32 --imgsz 640
"""

import os
import sys
import shutil
import zipfile
import argparse
import subprocess
from pathlib import Path

# ANSI Terminal Colors
C_BOLD_GREEN = "\033[1;92m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_MAGENTA = "\033[95m"
C_RED = "\033[91m"
C_RESET = "\033[0m"


def get_metric_color(val: float) -> str:
    if val >= 90.0: return C_BOLD_GREEN
    elif val >= 80.0: return C_GREEN
    elif val >= 65.0: return C_YELLOW
    elif val >= 45.0: return C_MAGENTA
    else: return C_RED


def mount_google_drive() -> Path:
    """Mounts Google Drive if running in Google Colab."""
    try:
        from google.colab import drive
        print("[*] Mounting Google Drive...")
        drive.mount('/content/drive', force_remount=False)
        drive_root = Path('/content/drive/MyDrive')
        print("  ✓ Google Drive successfully mounted at /content/drive/MyDrive")
        return drive_root
    except ImportError:
        print("[!] Not running inside Google Colab. Using local working directory as mock Drive.")
        mock_drive = Path("./mock_drive")
        mock_drive.mkdir(exist_ok=True)
        return mock_drive


def extract_dataset_to_colab_nvme(drive_root: Path, target_dir: Path = Path("/content/dataset_rail-drishti")):
    """Extracts raildrishti_colab.zip from Google Drive to local fast Colab NVMe SSD."""
    zip_path = drive_root / "raildrishti_colab.zip"
    
    # Check if dataset is already extracted
    if (target_dir / "images" / "train").exists() and len(list((target_dir / "images" / "train").glob("*.jpg"))) > 1000:
        print(f"  ✓ Dataset already extracted and ready at {target_dir}")
        return True
        
    if not zip_path.exists():
        # Also check current working directory
        if Path("raildrishti_colab.zip").exists():
            zip_path = Path("raildrishti_colab.zip")
        else:
            print(f"[!] Error: 'raildrishti_colab.zip' not found at {zip_path}")
            print("    Please upload 'raildrishti_colab.zip' to your Google Drive root folder (MyDrive) first.")
            return False

    print(f"[*] Extracting '{zip_path.name}' to local fast NVMe SSD ({target_dir})...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
        zip_ref.extractall(target_dir.parent)
        
    print(f"  ✓ Dataset extracted successfully! Ready for high-speed GPU training.")
    return True


class ColabAccuracyMonitor:
    """
    Minimal 3-line per-epoch accuracy reporter with persistent Drive backup.
    """
    TARGET_BOX_MAP = 85.0
    TARGET_SEG_MAP = 90.0
    TARGET_OVERALL_MAP = 85.0

    def __init__(self, drive_backup_dir: Path, drive_best_model_path: Path):
        self.best_map = 0.0
        self.best_epoch = 0
        self.drive_backup_dir = drive_backup_dir
        self.drive_best_model_path = drive_best_model_path
        self.drive_backup_dir.mkdir(parents=True, exist_ok=True)

    def on_fit_epoch_end(self, trainer):
        epoch = trainer.epoch + 1
        total_epochs = trainer.epochs
        metrics = getattr(trainer, "metrics", {}) or {}

        box_map50 = (metrics.get("metrics/mAP50(B)", 0.0) or 0.0) * 100.0
        seg_map50 = (metrics.get("metrics/mAP50(M)", 0.0) or 0.0) * 100.0
        overall_map50 = (box_map50 + seg_map50) / 2.0 if (box_map50 > 0 and seg_map50 > 0) else (box_map50 or seg_map50 or 0.0)

        loss_val = None
        if hasattr(trainer, "tloss") and trainer.tloss is not None:
            try:
                loss_val = float(trainer.tloss.mean()) if hasattr(trainer.tloss, "mean") else float(trainer.tloss)
            except Exception:
                loss_val = None

        # Check if new all-time best
        is_new_best = False
        if overall_map50 > self.best_map and overall_map50 > 1.0:
            self.best_map = overall_map50
            self.best_epoch = epoch
            is_new_best = True
            
            # Backup best model directly to Drive root
            if hasattr(trainer, "best") and os.path.exists(str(trainer.best)):
                try:
                    shutil.copy(str(trainer.best), str(self.drive_best_model_path))
                except Exception:
                    pass

        # Also backup latest checkpoint to Drive after EVERY epoch (guarantees resume safety)
        if hasattr(trainer, "last") and os.path.exists(str(trainer.last)):
            try:
                shutil.copy(str(trainer.last), str(self.drive_backup_dir / "last.pt"))
            except Exception:
                pass

        # Dynamic Colors
        c_overall = get_metric_color(overall_map50)
        c_seg = get_metric_color(seg_map50)
        c_box = get_metric_color(box_map50)
        c_best = get_metric_color(self.best_map)
        rst = C_RESET

        loss_str = f"{loss_val:.4f}" if loss_val is not None else "N/A"
        best_msg = f" (★ Backed up to Google Drive: {self.drive_best_model_path.name})" if is_new_best else ""

        # Minimal 3-line output
        print(f"\nEpoch [{epoch:02d}/{total_epochs:02d}] -> Current Accuracy: {c_overall}{overall_map50:.1f}%{rst} (Expected: ≥{self.TARGET_OVERALL_MAP:.0f}%) | Loss: {loss_str}")
        print(f"Metrics: Track Segm mAP: {c_seg}{seg_map50:.1f}%{rst} (Expected: ≥{self.TARGET_SEG_MAP:.0f}%) | Obstacle Box mAP: {c_box}{box_map50:.1f}%{rst} (Expected: ≥{self.TARGET_BOX_MAP:.0f}%)")
        print(f"Best Accuracy: {c_best}{self.best_map:.1f}%{rst} at Epoch {self.best_epoch}{best_msg}\n")


def run_colab_training(
    action: str = "auto",
    epochs: int = 60,
    batch_size: int = 32,
    imgsz: int = 640,
    model_name: str = "yolo11s-seg.pt"
):
    """
    Main training execution function for Google Colab Cloud GPUs.
    """
    # 1. Mount Drive
    drive_root = mount_google_drive()
    drive_ckpt_dir = drive_root / "drishti_checkpoints"
    drive_last_ckpt = drive_ckpt_dir / "last.pt"
    drive_best_model = drive_root / "RailDrishti.pt"

    # 2. Extract Dataset
    colab_dataset_dir = Path("/content/dataset_rail-drishti")
    if not extract_dataset_to_colab_nvme(drive_root, colab_dataset_dir):
        # Fallback to local path if not in colab
        colab_dataset_dir = Path("dataset_rail-drishti")

    # 3. Create or verify dataset YAML config for Colab
    colab_yaml_path = Path("/content/raildrishti_colab.yaml")
    yaml_content = f"""path: {colab_dataset_dir.resolve()}
train: images/train
val: images/val
names:
  0: Rail_Track_Bed
  1: Rail_Lines
  2: Person
  3: Car
  4: Truck
  5: Branch
  6: IronRod
  7: Barrel
  8: Boulder
  9: Jerrycan
"""
    with open(colab_yaml_path, "w") as f:
        f.write(yaml_content)

    # 4. Handle Action Selection (Fresh vs. Resume)
    if action == "auto":
        print("\n" + "=" * 70)
        print("       DRISHTI-KAVACH: GOOGLE COLAB TRAINING PIPELINE")
        print("=" * 70)
        has_checkpoint = drive_last_ckpt.exists()
        print(" Choose an action:")
        print("   [1] Start FRESH Training (From scratch with pretrained YOLO11 weights)")
        if has_checkpoint:
            print(f"   [2] RESUME Paused Training (Found previous checkpoint on Google Drive: {drive_last_ckpt})")
        else:
            print("   [2] Resume Training (No checkpoint currently found on Drive)")
            
        choice = input("\nEnter choice [1 or 2] (default: 1): ").strip()
        if choice == "2" and has_checkpoint:
            action = "resume"
        else:
            action = "fresh"

    from ultralytics import YOLO
    
    # 5. Initialize Model
    if action == "resume":
        if not drive_last_ckpt.exists():
            print(f"[!] Warning: Checkpoint not found at {drive_last_ckpt}. Starting fresh training instead.")
            model = YOLO(model_name)
            is_resume = False
        else:
            print(f"\n[*] Resuming training directly from Google Drive checkpoint: {drive_last_ckpt}...")
            # Copy to local fast NVMe
            local_resume_ckpt = Path("/content/last.pt")
            shutil.copy(str(drive_last_ckpt), str(local_resume_ckpt))
            model = YOLO(str(local_resume_ckpt))
            is_resume = True
    else:
        print(f"\n[*] Starting fresh training with base weights: {model_name}...")
        model = YOLO(model_name)
        is_resume = False

    # 6. Setup Monitor
    monitor = ColabAccuracyMonitor(drive_backup_dir=drive_ckpt_dir, drive_best_model_path=drive_best_model)
    model.add_callback("on_fit_epoch_end", monitor.on_fit_epoch_end)

    # 7. Launch Training
    if is_resume:
        model.train(resume=True)
    else:
        train_args = {
            "data": str(colab_yaml_path),
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": imgsz,
            "device": 0,           # Colab Cloud GPU (NVIDIA A100 / L4 / T4)
            "workers": 8,
            "project": "/content/runs",
            "name": "colab_raildrishti",
            "exist_ok": True,
            "amp": True,
            "optimizer": "AdamW",
            "lr0": 0.002,
            "lrf": 0.01,
            "cos_lr": True,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "patience": 15,
            "save": True,
            "save_period": 5,
            "val": True,
            "mosaic": 1.0,
            "mixup": 0.1,
            "fliplr": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.6,
            "hsv_v": 0.4
        }
        model.train(**train_args)

    print("\n" + "=" * 70)
    print(" 🎉 TRAINING FINISHED!")
    print("=" * 70)
    print(f" • Best model saved to Google Drive: {drive_best_model}")
    print(" • You can now download 'RailDrishti.pt' from your Google Drive into 'models/' locally.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Google Colab Training Pipeline")
    parser.add_argument("--action", type=str, default="auto", choices=["auto", "fresh", "resume"], help="Training mode")
    parser.add_argument("--epochs", type=int, default=60, help="Total epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (e.g. 32 for Colab GPU)")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolution (640 or 1024)")
    parser.add_argument("--model", type=str, default="yolo11s-seg.pt", help="Pretrained weights")
    args = parser.parse_args()

    run_colab_training(
        action=args.action,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        model_name=args.model
    )
