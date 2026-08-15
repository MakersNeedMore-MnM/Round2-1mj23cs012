"""
Drishti Kavach: Training Engine for Unified "RailDrishti" Model
Clean Terminal Output with Plain-English Epoch Progress Callbacks
"""

import os
import sys
import shutil
import warnings
import argparse

# 1. Suppress all non-essential Python and library warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["YOLO_VERBOSE"] = "False"

import torch
from ultralytics import YOLO


def on_epoch_end_callback(trainer):
    """
    Custom Ultralytics callback triggered at the end of every training epoch.
    Prints a clean, concise, plain-English evaluation of model performance.
    """
    epoch = trainer.epoch + 1
    total_epochs = trainer.epochs
    
    # Extract validation metrics safely
    metrics = trainer.metrics if hasattr(trainer, "metrics") else {}
    
    box_map50 = metrics.get("metrics/mAP50(B)", 0.0) * 100.0
    mask_map50 = metrics.get("metrics/mAP50(M)", 0.0) * 100.0
    
    # Combined score
    avg_score = (box_map50 + mask_map50) / 2.0 if mask_map50 > 0 else box_map50
    
    # Plain English status evaluation
    if epoch <= 3 and avg_score < 30.0:
        status = "Warming up (Model is learning initial railway features & track edges)"
        rating = "Initializing"
    elif avg_score < 50.0:
        status = "Progressing (Beginning to locate tracks and distinct obstacles)"
        rating = "Fair"
    elif avg_score < 75.0:
        status = "Good progress (Accurately localizing obstacles & track bed)"
        rating = "Good"
    elif avg_score < 88.0:
        status = "Strong performance (High accuracy across tracks & hazard classes)"
        rating = "Very Good"
    else:
        status = "Excellent performance! (Superb accuracy on tracks & small sabotage items)"
        rating = "Outstanding"
        
    next_action = f"Continuing to Epoch {epoch + 1}/{total_epochs}..." if epoch < total_epochs else "Training Complete!"
    
    print(f"\n[Epoch {epoch:2d}/{total_epochs:2d}] Box mAP: {box_map50:5.1f}% | Mask mAP: {mask_map50:5.1f}% | Rating: {rating:<11} | {status} | {next_action}")


def train_raildrishti(
    dataset_yaml: str = "configs/raildrishti_dataset.yaml",
    base_model: str = "yolo11s-seg.pt",
    epochs: int = 40,
    imgsz: int = 1024,
    batch_size: int = 8,
    device: str = None,
    output_model_path: str = "models/RailDrishti.pt"
):
    print("=" * 80)
    print(" 🛡️  DRISHTI KAVACH: TRAINING UNIFIED 'RailDrishti' MULTI-TASK MODEL")
    print("=" * 80)

    # Automatically select best hardware accelerator
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            dev_name = "Apple Silicon GPU (MPS)"
        elif torch.cuda.is_available():
            device = "0"
            dev_name = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
        else:
            device = "cpu"
            dev_name = "CPU"
    else:
        dev_name = device

    print(f" • Model Architecture:  {base_model} (Unified Multi-Task Segmentation & Detection)")
    print(f" • Target Resolution:   {imgsz}x{imgsz}")
    print(f" • Training Epochs:     {epochs}")
    print(f" • Batch Size:          {batch_size}")
    print(f" • Hardware Device:     {dev_name}")
    print(f" • Dataset Config:      {dataset_yaml}")
    print("=" * 80 + "\n")

    # Initialize model
    model = YOLO(base_model)

    # Attach our custom clean English progress callback
    model.add_callback("on_fit_epoch_end", on_epoch_end_callback)

    # Start training
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=8,
        name="RailDrishti_Training",
        save=True,
        save_period=5,
        patience=12,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.08,
        scale=0.25,
        fliplr=0.5,
        mosaic=0.7,
        val=True,
        verbose=False
    )

    # Save best checkpoint to models/RailDrishti.pt
    best_pt = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    if os.path.exists(best_pt):
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        shutil.copy(best_pt, output_model_path)
        print("\n" + "=" * 80)
        print(f" 🎉 SUCCESS: 'RailDrishti' model saved to: {output_model_path}")
        print("=" * 80)
    else:
        print(f"[!] Model saved at: {best_pt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unified RailDrishti Model")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (e.g. 8 or 16)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input resolution (e.g. 1024 or 640)")
    parser.add_argument("--model", type=str, default="yolo11s-seg.pt", help="Base YOLO weights")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', '0', 'cpu')")
    args = parser.parse_args()

    train_raildrishti(
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        base_model=args.model,
        device=args.device
    )
