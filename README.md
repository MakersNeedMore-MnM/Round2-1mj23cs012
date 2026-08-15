# 🛡️ Drishti Kavach: AI-Powered Railway Physical Obstacle & Track Clearance System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/YOLO-11_Segmentation-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Drishti Kavach** is an AI-powered visual perception enhancement for the Indian Railways' indigenous **Kavach (Automatic Train Protection / ATP)** system. 

While Kavach provides critical protection against SPAD (Signal Passed at Danger), rear-end collisions, and overspeeding via RFID tags and UHF radio, it lacks optical perception for **physical foreign obstacles on tracks**. Drishti Kavach bridges this gap by unifying **real-time track segmentation** with **multi-class obstacle detection** across both **Daylight RGB** and **Active Infrared (850nm) Night Vision CCTV** surveillance.

---

## 🌟 Key Features

1. **Unified "RailDrishti" Multi-Task Architecture (`RailDrishti.pt`)**:
   - Single forward pass (>45 FPS) simultaneously predicting Track Bed (`Rail_Track_Bed`), Rail Lines (`Rail_Lines`), and multi-class obstacle instances.
2. **24/7 All-Weather Day & Active IR CCTV Capability**:
   - Physics-based sensor simulation engine converting daylight footage into authentic 850nm Active IR CCTV with 1:1 ground-truth preservation.
3. **Class Taxonomy**:
   - **Track**: `Rail_Track_Bed` (0), `Rail_Lines` (1)
   - **Sabotage & Physical Debris**: `Branch` (2), `IronRod` (3), `Barrel` (4), `Boulder` (5), `Jerrycan` (6)
   - **Living & Dynamic Threats**: `Person` (7), `Cattle` (8), `Animal` (9), `Vehicle` (10)
4. **Spatial Clearance & Hazard Decision Engine**:
   - Classifies threats in real-time into **🔴 CRITICAL (In-Track)**, **🟡 WARNING (Near-Track Clearance Breach)**, and **🟢 SAFE (Off-Track)**.
5. **BiSeNetV2-Style High-Contrast HUD**:
   - Translucent Cyan-Blue track bed overlay, glowing Emerald Green rail lines, and crisp Red obstacle bounding boxes.

---

## 📁 Repository Structure

```
drishti-kavach/
├── configs/
│   └── raildrishti_dataset.yaml      # Multi-task dataset configuration
├── src/
│   ├── augmentation/
│   │   └── night_cctv_converter.py   # Physics-based Active IR (850nm) sensor converter
│   ├── preprocessing/
│   │   ├── unified_dataset_builder.py# High-res 1080p dataset builder & packager
│   │   └── verify_annotations.py     # Ground-truth HUD verification tool
│   ├── training/
│   │   └── train_raildrishti.py      # Unified model trainer (Apple Silicon MPS / CUDA)
│   ├── spatial_reasoning/            # Clearance envelope & threat decision engine
│   ├── visualization/                # BiSeNetV2-style visualizer & HUD renderer
│   └── pipeline/                     # Real-time multi-stream inference engine
├── requirements.txt                  # Pinned dependencies
└── .gitignore
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Create virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Synthesize Active IR Night CCTV Footage
```bash
# Convert all images to Active IR CCTV style
python src/augmentation/night_cctv_converter.py --convert-all --workers 8
```

### 3. Package Unified Dataset & Verify
```bash
# Build unified dataset with 1080p native ground-truth masks
python src/preprocessing/unified_dataset_builder.py

# Visually verify samples
python src/preprocessing/verify_annotations.py
```

### 4. Train the "RailDrishti" Unified Model
```bash
# Train on Apple Silicon GPU (MPS) or NVIDIA CUDA
python src/training/train_raildrishti.py --epochs 35 --batch 8 --imgsz 1024
```

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).
