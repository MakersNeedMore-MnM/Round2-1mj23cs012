# 🛡️ Drishti-Kavach (दृष्टि कवच)
### *Next-Generation Optical Automatic Train Protection (ATP) & Railway Clearance Perception Engine*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/Model-YOLO11m%20%7C%20BiSeNetV2-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Platform](https://img.shields.io/badge/Edge%20AI-Jetson%20%7C%20CUDA%20%7C%20Apple%20Silicon-success.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

---

## 🎬 Project Links

- 📽️ **Demo Video (YouTube):** [Watch Live Demonstration](https://youtu.be/4pEMPV4H1E8?si=QKoXsBVRQ_k91wxz)
- 📑 **Slide Deck (Presentation):** [View Google Slides / PPTX](https://drive.google.com/file/d/1rKYyHr4xYIWacUblTJ6O0RYRIFfB8CdH/view?usp=sharing)

---

## 📌 Executive Overview

**Drishti-Kavach** is an edge-deployable deep learning perception and geometric spatial reasoning system engineered to augment the **Indian Railways Kavach (TCAS - Train Collision Avoidance System)** network.

While traditional ATP systems rely on fixed track circuits, axle counters, and RFID transponders to prevent train-to-train collisions, they cannot detect sudden physical track hazards (fallen trees, boulders, deliberate sabotage items, stalled vehicles, or human trespassers).

**Drishti-Kavach** introduces an **autonomous forward-looking optical clearance envelope**:
- Decouples perception into **high-resolution semantic track segmentation** and **specialized physical hazard detection**.
- Operates under **dual-spectrum illumination** (Daylight RGB and Active 850nm Near-Infrared Night Vision).
- Uses **geometric polygon spatial reasoning** (via Shapely) to classify threats and trigger instant Automatic Train Protection (ATP) braking telemetry.

---

## 🏗️ System Architecture & Perception Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DRISHTI-KAVACH PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  [ Forward Optical Feed ]  ──►  [ Atmospheric Optical Enhancer ] (Adaptive CLAHE / DCP)    │
│  (Daylight RGB / 850nm NIR)                               │                                 │
│                                   ┌───────────────────────┴───────────────────────┐         │
│                                   ▼                                               ▼         │
│                  [ BiSeNetV2 Semantic Segmenter ]               [ YOLO11m Obstacle Detector ]│
│                  • Track Bed Ballast (Drivable Gauge)           • 8 Specialized Classes     │
│                  • Steel Rail Ribbons (Distance Vector)         • Sabotage & Hazard Items   │
│                                   │                                               │         │
│                                   └───────────────────────┬───────────────────────┘         │
│                                                           ▼                                 │
│                                [ Spatial Hazard & Vector Clearance Engine ]                 │
│                                • Shapely Footprint Polygon Intersection                     │
│                                • Dynamic Lateral Safety Buffer Analysis                     │
│                                • 3-Tier Threat Classification (CRITICAL / WARNING / CLEAR)  │
│                                                           │                                 │
│                                                           ▼                                 │
│                                [ Real-Time Railway Telemetry & Driver HUD ]                 │
│                                • Emergency Brake Trigger (ATP Interface)                    │
│                                • Low-Latency Dynamic Overlay Dashboard                      │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🌟 Key Technical Innovations

### 1. ⚡ Decoupled Dual-Engine Architecture
- **BiSeNetV2 Track Segmenter (~3.49M params):** Lightweight bilateral network with dedicated Detail & Semantic branches, achieving **>60 FPS** on edge accelerators for millimeter-accurate drivable gauge boundary estimation.
- **YOLO11m Hazard & Sabotage Detector:** Custom-trained 8-class railway hazard model optimized at 1024×1024 for extended forward lookahead distances.

### 2. 📐 Geometric Vector Spatial Reasoning
- Projects detected obstacle ground-contact footprints against segmented track polygons using vector geometry (`shapely`).
- **3-Tier Decision Engine:**
  - 🔴 **CRITICAL (In-Track):** Direct overlap with active track bed/rails $\rightarrow$ Instantly triggers ATP emergency braking telemetry.
  - 🟡 **WARNING (Near-Track):** Hazard within dynamic lateral safety clearance buffer $\rightarrow$ Issues high-priority driver warning.
  - 🟢 **CLEAR (Off-Track):** Detected objects outside clearance envelope $\rightarrow$ Logged without false alarm brake triggers.

### 3. 🌙 Dual-Spectrum & All-Weather Robustness
- **Daylight RGB + 850nm Active NIR:** Physics-based night vision support with spotlight beam cone modeling and radial intensity decay.
- **Atmospheric Enhancer:** Integrates Contrast Limited Adaptive Histogram Equalization (CLAHE) and Dark Channel Prior (DCP) defogging for dense fog, rain, and heavy glare conditions.

---

## 🎯 Perception Taxonomy

### 🛤️ Semantic Segmentation (3 Classes)
| Class ID | Target | Purpose in Safety Envelope |
| :---: | :--- | :--- |
| **0** | `Background` | Non-rail surroundings, catenary, terrain, sky |
| **1** | `Track Bed` | Ballast drivable gauge footprint (primary clearance boundary) |
| **2** | `Rail Lines` | Running steel rails for track horizon & center vectoring |

### ⚠️ Physical Hazard & Sabotage Detection (8 Classes)
| Class ID | Class Name | Target Description | Hazard Severity |
| :---: | :--- | :--- | :---: |
| **0** | `Person` | Pedestrians, trespassers, track maintenance workers | **High** |
| **1** | `Car` | Passenger motor vehicles stalled at level crossings | **Critical** |
| **2** | `Truck` | Commercial trucks, tractors, heavy construction machinery | **Critical** |
| **3** | `Branch` | Fallen tree limbs, storm foliage debris | **High** |
| **4** | `IronRod` | Deliberate sabotage items (placed rails, rods, fishplates) | **Critical** |
| **5** | `Boulder` | Landslide rocks, placed stones, heavy ballast debris | **Critical** |
| **6** | `Barrel` | Metal/plastic drums and containers on track | **High** |
| **7** | `Jerrycan` | Flammable canisters and hazardous containers | **High** |

---

## 📊 Benchmark Performance Summary

| Perception Subsystem | Architecture / Model | Resolution | Parameters | Benchmark Metric | Result |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Track Bed Segmentation** | BiSeNetV2 (Universal) | 512×1024 | ~3.49M | Track Bed IoU | **89.5%** |
| **Rail Line Segmentation** | BiSeNetV2 (Universal) | 512×1024 | ~3.49M | Rail Lines IoU | **68.5%** |
| **Overall Segmentation** | BiSeNetV2 (Universal) | 512×1024 | ~3.49M | Mean IoU (mIoU) | **86.0%** |
| **Obstacle Detection** | YOLO11m (8-Class) | 1024×1024 | ~20.1M | mAP@50 | **60.8%** |
| **Inference Pipeline Speed** | Decoupled Dual-Engine | Combined | Total Engine | End-to-End Latency | **< 22ms (>45 FPS)** |

---

## 📁 Repository Structure

```
drishti-kavach/
├── models/                           # Trained deep learning checkpoints & ONNX models
│   ├── RailDrishti_Seg_BiSeNetV2.pth # Universal Track Segmenter Checkpoint (86.0% mIoU)
│   ├── RailDrishti_Det_YOLO11m.pt    # Custom 8-Class Railway Sabotage Detector
│   └── *.onnx                        # Production-ready edge inference ONNX exports
├── src/
│   ├── 1_preprocessing/              # Dataset converters (RailSem19, UAV-RSOD, 8-Class)
│   ├── 2_models/                     # Model definitions (BiSeNetV2 bilateral network)
│   ├── 3_kaggle_train/               # Cloud GPU distributed training pipelines
│   ├── 4_local_train/                # Local training & fine-tuning scripts
│   ├── 5_spatial_reasoning/          # Shapely polygon clearance & ATP decision engine
│   ├── 6_visualization/              # Railway HUD telemetry compositor
│   └── 7_pipeline/                   # Drishti-Kavach master engine & weather enhancer
├── test_samples/                     # Evaluation test images & videos (Day/Night)
├── accuracy_metrics_test.py          # Automated model benchmark & verification suite
├── arducam_tester.py                 # Live Arducam 0506 Day/Night camera calibration tool
├── camera_tester.py                  # Live video capture probe utility
├── requirements.txt                  # Python package requirements
└── main.py                           # Master CLI & GUI inference entry point
```

## 🛠️ Technology Stack

- **Deep Learning Frameworks:** PyTorch, Torchvision, Ultralytics YOLO11, ONNX Runtime
- **Spatial Vector Geometry:** Shapely, OpenCV, NumPy
- **Image Enhancement:** SciPy, Scikit-Image, Adaptive CLAHE, Dark Channel Prior (DCP)
- **Edge Deployment Targets:** NVIDIA Jetson (Orin / Xavier), Apple Silicon (MPS), CUDA Desktops

---

## 👥 Authors & Team

**Developed by Team Git Commit**  
*From India, For India* 🇮🇳
