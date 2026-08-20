"""
Drishti-Kavach: Local BiSeNetV2 Semantic Segmentation Trainer (Mac Apple Silicon MPS / CUDA / CPU)

Trains BiSeNetV2 on `dataset_segmentation/` with real-time validation mIoU tracking.

Usage:
  # 1. Quick test run on 100 samples (Fast sanity check):
  python src/4_local_train/train_local.py --max-samples 100 --epochs 3 --batch-size 4

  # 2. Standard local training:
  python src/4_local_train/train_local.py --epochs 20 --batch-size 4 --img-h 512 --img-w 1024
"""

import os
import sys
import glob
import time
import random
import argparse
import importlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Dynamic import for directory starting with a digit
_models = importlib.import_module("src.2_models.bisenetv2")
BiSeNetV2 = _models.BiSeNetV2


# -----------------------------------------------------------------------------
# 1. Dataset Loader with Online Augmentations
# -----------------------------------------------------------------------------
class RailwaySegmentationDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", img_size=(512, 1024), is_train: bool = True, max_samples: int = None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_h, self.img_w = img_size
        self.is_train = is_train

        self.img_paths = sorted(glob.glob(str(self.root_dir / "images" / split / "*.jpg")))
        self.mask_paths = [
            p.replace("/images/", "/masks/").replace(".jpg", ".png")
            for p in self.img_paths
        ]

        if max_samples and max_samples < len(self.img_paths):
            random.seed(42)
            indices = list(range(len(self.img_paths)))
            random.shuffle(indices)
            sel = indices[:max_samples]
            self.img_paths = [self.img_paths[i] for i in sel]
            self.mask_paths = [self.mask_paths[i] for i in sel]

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.img_paths[idx])
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read: {self.img_paths[idx]}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Failed to read: {self.mask_paths[idx]}")

        # Resize to target resolution
        img_resized = cv2.resize(img_rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

        # Online Augmentations for training
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                img_resized = cv2.flip(img_resized, 1)
                mask_resized = cv2.flip(mask_resized, 1)

            # Random Brightness & Contrast
            if random.random() > 0.5:
                alpha = 1.0 + random.uniform(-0.15, 0.15)
                beta = random.uniform(-15, 15)
                img_resized = np.clip(alpha * img_resized + beta, 0, 255).astype(np.uint8)

        # Normalize to standard tensor
        img_norm = (img_resized / 255.0 - self.mean) / self.std
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_resized).long()

        return img_tensor, mask_tensor


# -----------------------------------------------------------------------------
# 2. Loss Functions (OHEM Cross-Entropy + Auxiliary Boosters)
# -----------------------------------------------------------------------------
class OhemCrossEntropy(nn.Module):
    def __init__(self, thresh=0.7, min_kept=100000, ignore_index=255):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, predict, target):
        b, c, h, w = predict.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(predict, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1.0)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)
        loss = self.criterion(predict, target)
        return loss[valid_mask.view(b, h, w)].mean()


class BiSeNetLoss(nn.Module):
    def __init__(self):
        super(BiSeNetLoss, self).__init__()
        self.crit = OhemCrossEntropy(thresh=0.7, min_kept=100000)

    def forward(self, preds, target):
        if isinstance(preds, tuple):
            main_out, aux2, aux3, aux4, aux5 = preds
            l_main = self.crit(main_out, target)
            l_aux2 = self.crit(aux2, target)
            l_aux3 = self.crit(aux3, target)
            l_aux4 = self.crit(aux4, target)
            l_aux5 = self.crit(aux5, target)
            return l_main + 0.4 * (l_aux2 + l_aux3 + l_aux4 + l_aux5)
        return self.crit(preds, target)


# -----------------------------------------------------------------------------
# 3. mIoU Evaluator Matrix
# -----------------------------------------------------------------------------
class Evaluator:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def add_batch(self, gt, pred):
        mask = (gt >= 0) & (gt < self.num_classes)
        label = self.num_classes * gt[mask].astype(int) + pred[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)

    def evaluate(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection
        ious = intersection / np.maximum(union, 1e-7)
        miou = np.nanmean(ious)
        return miou, ious

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))


# -----------------------------------------------------------------------------
# 4. Main Training Pipeline
# -----------------------------------------------------------------------------
def train_local(
    dataset_dir: str = "dataset_segmentation",
    save_dir: str = "models",
    img_h: int = 512,
    img_w: int = 1024,
    batch_size: int = 4,
    epochs: int = 20,
    lr: float = 0.005,
    num_workers: int = 2,
    max_samples: int = None,
    device_choice: str = "auto"
):
    # Select Device
    if device_choice == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_choice)

    out_models_dir = Path(save_dir)
    out_models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" 🛡️ DRISHTI-KAVACH: LOCAL BISENETV2 TRAINING PIPELINE")
    print("=" * 75)
    print(f" • Device Selected:    {device}")
    print(f" • Dataset Path:       {Path(dataset_dir).resolve()}")
    print(f" • Target Resolution:  {img_w}x{img_h}")
    print(f" • Batch Size:         {batch_size}")
    print(f" • Total Epochs:       {epochs}")
    print(f" • Learning Rate:      {lr}")
    print(f" • Checkpoint Path:    {out_models_dir.resolve()}")
    print("=" * 75)

    # Prepare Datasets & DataLoaders
    img_size = (img_h, img_w)
    train_dataset = RailwaySegmentationDataset(dataset_dir, split="train", img_size=img_size, is_train=True, max_samples=max_samples)
    val_dataset = RailwaySegmentationDataset(dataset_dir, split="val", img_size=img_size, is_train=False, max_samples=max_samples // 5 if max_samples else None)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print(f"[!] Error: No data found in {dataset_dir}. Run prep_railsem19_segmentation.py first.")
        return

    print(f"[+] Loaded {len(train_dataset)} Train samples and {len(val_dataset)} Validation samples.")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    # Initialize Model, Loss, Optimizer, Scheduler
    model = BiSeNetV2(num_classes=3, is_training=True).to(device)
    criterion = BiSeNetLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)
    evaluator = Evaluator(num_classes=3)

    best_miou = 0.0

    print("\n🚀 Starting Training Loop...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{epochs}]", dynamic_ncols=True)
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}"
            })

        avg_train_loss = total_loss / len(train_loader)
        epoch_duration = time.time() - start_time

        # Validation Phase
        model.eval()
        evaluator.reset()
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc="Validating", leave=False, dynamic_ncols=True):
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                evaluator.add_batch(masks.numpy(), preds)

        miou, class_ious = evaluator.evaluate()

        print(f"\n📊 Epoch [{epoch:02d}/{epochs}] Summary ({epoch_duration:.1f}s):")
        print(f" • Train Loss:       {avg_train_loss:.4f}")
        print(f" • Validation mIoU:  {miou * 100:.2f}%")
        print(f"   - Background IoU: {class_ious[0] * 100:.2f}%")
        print(f"   - Track Bed IoU:  {class_ious[1] * 100:.2f}%")
        print(f"   - Rail Lines IoU: {class_ious[2] * 100:.2f}%")

        # Save Best Model Checkpoint
        if miou > best_miou:
            best_miou = miou
            best_pth = out_models_dir / "best_bisenetv2_local.pth"
            torch.save(model.state_dict(), str(best_pth))
            print(f" ⭐ NEW BEST MODEL SAVED: {best_pth} (mIoU: {best_miou * 100:.2f}%)")

        last_pth = out_models_dir / "last_bisenetv2_local.pth"
        torch.save(model.state_dict(), str(last_pth))
        print("-" * 75)

    print("\n" + "=" * 75)
    print(f" 🎉 LOCAL TRAINING COMPLETE! Best Validation mIoU: {best_miou * 100:.2f}%")
    print(f" • Best Weights: {out_models_dir / 'best_bisenetv2_local.pth'}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local BiSeNetV2 Trainer for Drishti-Kavach")
    parser.add_argument("--dataset-dir", type=str, default="dataset_segmentation", help="Path to segmentation dataset")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--img-h", type=int, default=512, help="Input image height")
    parser.add_argument("--img-w", type=int, default=1024, help="Input image width")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Initial learning rate")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for fast local test")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"], help="Hardware device")

    args = parser.parse_args()

    train_local(
        dataset_dir=args.dataset_dir,
        save_dir=args.save_dir,
        img_h=args.img_h,
        img_w=args.img_w,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        device_choice=args.device
    )
