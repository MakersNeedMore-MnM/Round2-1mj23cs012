"""
Drishti Kavach: Quantitative Model Accuracy & Benchmark Evaluation Suite

Evaluates:
  1. BiSeNetV2 Semantic Track & Rail Segmentation Engine (IoU, mIoU, Dice, Pixel Acc, Latency)
  2. YOLO11m Railway Obstacle & Sabotage Detector (mAP@50, mAP@50-95, Precision, Recall)
  3. Domain & Spectrum Breakdown (RailSem19 Locomotive vs Indian UAV, Daylight vs Active NIR Night)

CLI Usage:
  python evaluate_models.py
  python evaluate_models.py --seg-model models/RailDrishti_Seg_Universal.pth --max-samples 200
  python evaluate_models.py --output-report outputs/accuracy_benchmark_report.txt
"""

import os
import sys
import time
import argparse
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib
_models = importlib.import_module("src.2_models.bisenetv2")
BiSeNetV2 = _models.BiSeNetV2


class SegmentationEvaluator:
    """Computes comprehensive pixel-level segmentation metrics."""
    def __init__(self, num_classes: int = 3, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or ["Background", "Track_Bed", "Rail_Lines"]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def add_batch(self, gt: np.ndarray, pred: np.ndarray):
        mask = (gt >= 0) & (gt < self.num_classes)
        label = self.num_classes * gt[mask].astype(int) + pred[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)

    def compute_metrics(self) -> Dict:
        cm = self.confusion_matrix
        tp = np.diag(cm).astype(float)
        sum_row = np.sum(cm, axis=1).astype(float)  # Ground Truth counts
        sum_col = np.sum(cm, axis=0).astype(float)  # Prediction counts
        
        # 1. Intersection over Union (IoU)
        union = sum_row + sum_col - tp
        ious = np.divide(tp, np.maximum(union, 1e-7))
        miou = np.nanmean(ious)
        
        # 2. Pixel Accuracy (Overall PA)
        total_pixels = np.sum(cm)
        pixel_acc = np.sum(tp) / max(total_pixels, 1)
        
        # 3. Class Accuracy (Recall per class) & Mean Pixel Accuracy (MPA)
        recalls = np.divide(tp, np.maximum(sum_row, 1e-7))
        mpa = np.nanmean(recalls)
        
        # 4. Precision per class
        precisions = np.divide(tp, np.maximum(sum_col, 1e-7))
        
        # 5. Dice / F1 Score
        dice = np.divide(2 * tp, np.maximum(sum_row + sum_col, 1e-7))
        m_dice = np.nanmean(dice)
        
        return {
            "miou": miou,
            "ious": ious,
            "pixel_acc": pixel_acc,
            "mpa": mpa,
            "recalls": recalls,
            "precisions": precisions,
            "dice": dice,
            "m_dice": m_dice,
            "confusion_matrix": cm
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)


def evaluate_segmentation_subset(
    model: nn.Module,
    image_paths: List[str],
    device: torch.device,
    desc: str = "Evaluating",
    max_samples: int = None
) -> Tuple[Dict, float]:
    """Runs inference and evaluates metrics on a list of image paths."""
    if max_samples and max_samples < len(image_paths):
        image_paths = image_paths[:max_samples]
        
    evaluator = SegmentationEvaluator(3, ["Background", "Track_Bed", "Rail_Lines"])
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    latencies = []
    
    for img_p in tqdm(image_paths, desc=desc, unit="img"):
        mask_p = img_p.replace("/images/", "/masks/").replace(".jpg", ".png")
        if not os.path.exists(mask_p):
            continue
            
        raw_bgr = cv2.imread(img_p)
        gt_mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)
        if raw_bgr is None or gt_mask is None:
            continue
            
        h_orig, w_orig = raw_bgr.shape[:2]
        
        # Preprocess
        img_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (1024, 512), interpolation=cv2.INTER_LINEAR)
        norm = (img_resized / 255.0 - mean) / std
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        t0 = time.time()
        with torch.no_grad():
            logits = model(tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0)  # ms
        
        pred_full = cv2.resize(pred.astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        evaluator.add_batch(gt_mask, pred_full)
        
    metrics = evaluator.compute_metrics()
    avg_latency_ms = float(np.mean(latencies)) if latencies else 0.0
    return metrics, avg_latency_ms


def generate_accuracy_report(
    seg_model_path: str = "models/RailDrishti_Seg_Universal.pth",
    det_model_path: str = "models/best_yolo11m_raildrishti.pt",
    output_report_path: str = "outputs/accuracy_benchmark_report.txt",
    max_samples: int = None
):
    os.makedirs(os.path.dirname(output_report_path) or ".", exist_ok=True)
    report_lines = []
    
    def log(msg: str = ""):
        print(msg)
        report_lines.append(msg)

    log("=" * 80)
    log(" 🛡️  DRISHTI-KAVACH: QUANTITATIVE ACCURACY & BENCHMARK REPORT")
    log("=" * 80)
    log(f" • Generated At:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f" • Segmentation Model: {seg_model_path}")
    log(f" • Detection Model:    {det_model_path}")
    log("=" * 80)

    # 1. Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dev_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dev_name = "Apple Silicon GPU (MPS)"
    else:
        device = torch.device("cpu")
        dev_name = "CPU"
    log(f" • Execution Device:   {dev_name}\n")

    # =========================================================================
    # PART 1: TRACK & RAIL SEMANTIC SEGMENTATION EVALUATION (BiSeNetV2)
    # =========================================================================
    log("-" * 80)
    log(" 🚆 PART 1: TRACK & RAIL SEMANTIC SEGMENTATION EVALUATION (BiSeNetV2)")
    log("-" * 80)

    if not os.path.exists(seg_model_path):
        log(f"[!] Warning: Segmentation weights not found at: {seg_model_path}")
    else:
        model_seg = BiSeNetV2(num_classes=3, is_training=False).to(device)
        sd = torch.load(seg_model_path, map_location=device)
        infer_sd = {k: v for k, v in sd.items() if not k.startswith("aux")}
        model_seg.load_state_dict(infer_sd, strict=False)
        model_seg.eval()
        
        num_params = sum(p.numel() for p in model_seg.parameters())
        log(f"[+] BiSeNetV2 Architecture Loaded: {num_params:,} Trainable Parameters (~{num_params/1e6:.2f}M)")

        # Collect Validation Subsets
        rs19_all = sorted(glob.glob("dataset_segmentation/images/val/*.jpg"))
        rs19_day = [p for p in rs19_all if "_day" in p]
        rs19_night = [p for p in rs19_all if "_night" in p]
        
        uav_all = sorted(glob.glob("dataset_segmentation_uav_v1/images/val/*.jpg"))
        uav_day = [p for p in uav_all if "_day" in p]
        uav_night = [p for p in uav_all if "_night" in p]
        
        combined_all = rs19_all + uav_all

        # Evaluate Combined Universal Set
        if combined_all:
            log(f"\n[*] Evaluating Combined Universal Validation Set ({len(combined_all)} Total Frames)...")
            m_comb, lat_comb = evaluate_segmentation_subset(model_seg, combined_all, device, "Universal Set", max_samples)
            fps_comb = 1000.0 / max(lat_comb, 0.001)

            log("\n" + "=" * 55)
            log(" 📊 COMBINED UNIVERSAL SEGMENTATION METRICS")
            log("=" * 55)
            log(f" • Overall Mean IoU (mIoU):       {m_comb['miou']*100:6.2f}%")
            log(f" • Overall Pixel Accuracy (PA):   {m_comb['pixel_acc']*100:6.2f}%")
            log(f" • Mean Pixel Accuracy (MPA):     {m_comb['mpa']*100:6.2f}%")
            log(f" • Mean Dice / F1 Score:          {m_comb['m_dice']*100:6.2f}%")
            log(f" • Inference Latency:             {lat_comb:6.2f} ms ({fps_comb:.1f} FPS)")
            log("-" * 55)
            log(" Per-Class Breakdown:")
            log(f"   [0] Background (Non-Rail):     {m_comb['ious'][0]*100:6.2f}% IoU | Recall: {m_comb['recalls'][0]*100:5.2f}% | Precision: {m_comb['precisions'][0]*100:5.2f}%")
            log(f"   [1] Track Bed (Drivable Ballast):{m_comb['ious'][1]*100:6.2f}% IoU | Recall: {m_comb['recalls'][1]*100:5.2f}% | Precision: {m_comb['precisions'][1]*100:5.2f}%")
            log(f"   [2] Rail Lines (Steel Rails):  {m_comb['ious'][2]*100:6.2f}% IoU | Recall: {m_comb['recalls'][2]*100:5.2f}% | Precision: {m_comb['precisions'][2]*100:5.2f}%")
            log("=" * 55)

        # Domain Breakdown
        log("\n" + "-" * 80)
        log(" 🌐 DOMAIN & ILLUMINATION BREAKDOWN (Locomotive vs Indian UAV, Day vs Night)")
        log("-" * 80)
        
        benchmarks = [
            ("Locomotive Cab (RailSem19 All)", rs19_all),
            ("Locomotive Cab (Daylight RGB)", rs19_day),
            ("Locomotive Cab (Active 850nm NIR Night)", rs19_night),
            ("Indian Infrastructure (UAV-RSOD All)", uav_all),
            ("Indian Infrastructure (Daylight RGB)", uav_day),
            ("Indian Infrastructure (Active 850nm NIR Night)", uav_night),
        ]

        log(f"{'Domain / Condition':<40} | {'Frames':<6} | {'mIoU (%)':<8} | {'Track Bed (%)':<13} | {'Rails (%)':<9} | {'FPS':<6}")
        log("-" * 92)

        for label, img_list in benchmarks:
            if img_list:
                m_sub, lat_sub = evaluate_segmentation_subset(model_seg, img_list, device, label, max_samples=max_samples)
                fps_sub = 1000.0 / max(lat_sub, 0.001)
                log(f"{label:<40} | {len(img_list):<6} | {m_sub['miou']*100:6.2f}%  | {m_sub['ious'][1]*100:11.2f}%   | {m_sub['ious'][2]*100:7.2f}%  | {fps_sub:5.1f}")
        log("-" * 92)

    # =========================================================================
    # PART 2: 8-CLASS RAILWAY OBSTACLE DETECTION EVALUATION (YOLO11m)
    # =========================================================================
    log("\n" + "-" * 80)
    log(" ⚠️  PART 2: 8-CLASS PHYSICAL OBSTACLE & SABOTAGE DETECTOR (YOLO11m)")
    log("-" * 80)

    yaml_path = "dataset_detection/detection_data.yaml"
    if not os.path.exists(det_model_path):
        log(f"[*] Obstacle weights '{det_model_path}' not found locally yet.")
        log("    (Once YOLO11m Kaggle training finishes, place weights in 'models/' to include mAP metrics).")
    elif not os.path.exists(yaml_path):
        log(f"[!] Dataset config '{yaml_path}' not found.")
    else:
        from ultralytics import YOLO
        log(f"[+] Loading YOLO11m Obstacle Detector from: {det_model_path}")
        det_model = YOLO(det_model_path)
        
        log("[*] Running Ultralytics COCO Validation Evaluation...")
        val_res = det_model.val(data=yaml_path, imgsz=1024, split="val", verbose=False)
        
        map50 = val_res.box.map50 * 100
        map50_95 = val_res.box.map * 100
        mp = val_res.box.mp * 100
        mr = val_res.box.mr * 100
        
        log("\n" + "=" * 55)
        log(" 📊 YOLO11m 8-CLASS OBSTACLE DETECTION METRICS")
        log("=" * 55)
        log(f" • Mean Average Precision (mAP@50):    {map50:6.2f}%")
        log(f" • Mean Average Precision (mAP@50-95): {map50_95:6.2f}%")
        log(f" • Precision (P):                      {mp:6.2f}%")
        log(f" • Recall (R):                         {mr:6.2f}%")
        log("-" * 55)
        log(" Per-Class AP@50 Breakdown:")
        
        class_names = ["Person", "Car", "Truck", "Branch", "IronRod", "Boulder", "Barrel", "Jerrycan"]
        if hasattr(val_res.box, "maps") and val_res.box.maps is not None:
            for idx, cname in enumerate(class_names):
                if idx < len(val_res.box.maps):
                    cls_ap = val_res.box.maps[idx] * 100
                    log(f"   [{idx}] {cname:<12}: {cls_ap:6.2f}% AP@50")
        log("=" * 55)

    # Save to Text Report
    with open(output_report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print("\n" + "=" * 80)
    print(f" 📄 ACCURACY REPORT SAVED SUCCESSFULLY TO: {output_report_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Drishti-Kavach Models")
    parser.add_argument("--seg-model", type=str, default="models/RailDrishti_Seg_Universal.pth", help="BiSeNetV2 weights")
    parser.add_argument("--det-model", type=str, default="models/best_yolo11m_raildrishti.pt", help="YOLO11m weights")
    parser.add_argument("--output-report", type=str, default="outputs/accuracy_benchmark_report.txt", help="Report output file")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per subset for fast evaluation")

    args = parser.parse_args()

    generate_accuracy_report(
        seg_model_path=args.seg_model,
        det_model_path=args.det_model,
        output_report_path=args.output_report,
        max_samples=args.max_samples
    )
