"""
Drishti Kavach: Training Engine for Unified "RailDrishti" Model
"""

import os
import argparse
import shutil
import torch
from ultralytics import YOLO


def train_raildrishti(
    dataset_yaml: str = "configs/raildrishti_dataset.yaml",
    base_model: str = "yolo11s-seg.pt",
    epochs: int = 40,
    imgsz: int = 1024,
    batch_size: int = 8,
    device: str = None,
    output_model_path: str = "models/RailDrishti.pt"
):
    print("=" * 75)
    print(" DRISHTI KAVACH: TRAINING UNIFIED 'RailDrishti' MULTI-TASK MODEL")
    print("=" * 75)

    # Automatically select best available hardware accelerator
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            print("[+] Using Apple Silicon GPU Acceleration (MPS)")
        elif torch.cuda.is_available():
            device = "0"
            print(f"[+] Using NVIDIA CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("[!] No GPU accelerator detected, using CPU")

    print(f"Base Pretrained Architecture: {base_model}")
    print(f"Dataset Configuration:        {dataset_yaml}")
    print(f"Target Image Size:            {imgsz}x{imgsz}")
    print(f"Epochs:                       {epochs}")
    print(f"Batch Size:                   {batch_size}")
    print(f"Compute Device:               {device}")
    print("=" * 75)

    # Initialize model with pre-trained weights
    model = YOLO(base_model)

    # Train the unified model
    results = model.train(
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
        val=True
    )

    # Locate and save the best checkpoint to models/RailDrishti.pt
    best_pt = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    if os.path.exists(best_pt):
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        shutil.copy(best_pt, output_model_path)
        print("\n" + "=" * 75)
        print(f"[+] SUCCESS: Final trained model saved to: {output_model_path}")
        print("=" * 75)
    else:
        print(f"[!] Checkpoint not found at {best_pt}, check training logs.")

    # Validate final model on the validation split
    print("\n[+] Evaluating RailDrishti on Validation Set...")
    val_model = YOLO(output_model_path if os.path.exists(output_model_path) else best_pt)
    metrics = val_model.val(data=dataset_yaml, imgsz=imgsz, device=device)
    print("\n[+] Validation Metrics Completed!")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unified RailDrishti Model")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (e.g. 8 or 16)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input resolution (e.g. 1024 or 640)")
    parser.add_argument("--model", type=str, default="yolo11s-seg.pt", help="Base YOLO model weights (e.g. yolo11s-seg.pt, yolo11m-seg.pt)")
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'mps', '0', 'cpu'")
    args = parser.parse_args()

    train_raildrishti(
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        base_model=args.model,
        device=args.device
    )
