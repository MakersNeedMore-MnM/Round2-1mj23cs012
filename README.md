# 🛡️ Drishti-Kavach (दृष्टि कवच)
### *Next-Generation Optical Automatic Train Protection (ATP) & Railway Clearance Perception Engine*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/Model-YOLO11m%20%7C%20BiSeNetV2-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Platform](https://img.shields.io/badge/Edge%20AI-Jetson%20%7C%20CUDA%20%7C%20Apple%20Silicon-success.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)


---

## 📌 Executive Overview

**Drishti-Kavach** is an edge-deployable deep learning perception and geometric spatial reasoning system engineered to augment the **Indian Railways Kavach (TCAS - Train Collision Avoidance System)** network.

While traditional ATP systems rely on fixed track circuits, axle counters, and RFID transponders to prevent Signal Passed at Danger (SPAD) and train-to-train collisions, they cannot detect sudden **physical track hazards** (fallen trees, boulders, deliberate sabotage items, stalled vehicles, or human trespassers).

**Drishti-Kavach** introduces an **autonomous forward-looking optical clearance envelope**:
- **Decoupled Dual Perception:** High-resolution semantic track segmentation and fine-grained physical hazard localization run in synchronized parallel pipelines.
- **Dual-Spectrum Illumination:** 24/7 day and night operation supporting Daylight RGB and Active 850nm Near-Infrared (NIR) Night Vision.
- **Geometric Vector Spatial Reasoning:** Ground-contact footprint projection against segmented track polygons (via `Shapely`) for instant 3-tier threat classification and Automatic Train Protection (ATP) emergency braking telemetry.

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
│                                • Low-Latency Dynamic Overlay Telemetry                      │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🌟 Key Technical Innovations

### 1. ⚡ Decoupled Dual-Engine Architecture
- **BiSeNetV2 Track Segmenter (~3.49M params):** Lightweight bilateral network featuring dedicated Detail and Semantic branches, achieving **>60 FPS** on edge accelerators for millimeter-accurate drivable gauge boundary estimation.
- **YOLO11m Hazard & Sabotage Detector:** Custom-trained 8-class railway hazard model optimized at 1024×1024 resolution for extended forward lookahead distances.

### 2. 📐 Geometric Vector Spatial Reasoning
- Projects detected obstacle ground-contact footprints against segmented track polygons using vector geometry (`shapely`).
- **3-Tier Decision Engine:**
  - 🔴 **CRITICAL (In-Track):** Direct overlap with active track bed/rails $\rightarrow$ Instantly triggers ATP emergency braking telemetry.
  - 🟡 **WARNING (Near-Track):** Hazard within dynamic lateral safety clearance buffer $\rightarrow$ Issues high-priority driver audio-visual alert.
  - 🟢 **CLEAR (Off-Track):** Detected objects outside structural clearance envelope $\rightarrow$ Logged to telemetry without false alarm brake triggers.

### 3. 🌙 Dual-Spectrum & All-Weather Robustness
- **Daylight RGB + Active 850nm NIR:** Physics-based night vision support with spotlight beam cone modeling and radial intensity decay.
- **Atmospheric Enhancer:** Integrates Contrast Limited Adaptive Histogram Equalization (CLAHE) and Dark Channel Prior (DCP) defogging for dense fog, rain, and heavy glare conditions.

---

## 🎯 Perception Taxonomy

### 🛤️ Semantic Segmentation (3 Classes)
| Class ID | Target | Purpose in Safety Envelope | Benchmark IoU |
| :---: | :--- | :--- | :---: |
| **0** | `Background` | Non-rail surroundings, catenary, terrain, sky | — |
| **1** | `Track Bed` | Ballast drivable gauge footprint (primary clearance boundary) | **89.5%** |
| **2** | `Rail Lines` | Running steel rails for track horizon & center vectoring | **68.5%** |

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
| **Inference Pipeline Speed** | Decoupled Dual-Engine | 1080p Stream | Total Engine | End-to-End Latency | **< 22ms (>45 FPS)** |

---

## 🚀 Quickstart & Usage

### 1. Clone & Setup Environment
```bash
git clone https://github.com/alvinxsonny/drishti-kavach.git
cd drishti-kavach
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Master Perception Engine
```bash
# Run inference on live camera feed or test video
python main.py
```

### 3. Run Benchmark Verification & Sensor Calibration
```bash
# Run accuracy metrics test suite
python accuracy_metrics_test.py

# Live camera probe test
python camera_tester.py

# Arducam Day/Night NIR sensor calibration
python arducam_tester.py
```

### 4. Interactive Web Showcase
Open `index.html` in any web browser to view the interactive minimalist showcase with full visual outputs and infinite-loop demonstration feeds.

---

## 🛠️ Technology Stack

- **Deep Learning Frameworks:** PyTorch, Torchvision, Ultralytics YOLO11, ONNX Runtime
- **Spatial Vector Geometry:** Shapely, OpenCV, NumPy
- **Image Enhancement:** SciPy, Scikit-Image, Adaptive CLAHE, Dark Channel Prior (DCP)
- **Edge Deployment Targets:** NVIDIA Jetson (Orin / Xavier), Apple Silicon (MPS), CUDA Desktops

---

## 👥 Authors & Team

**Developed by Team Git Commit**  
- **Alvin Sonny** — Team Lead
- **Gowri Krishnan Nair** — Team Member
- **Ananya Sanjiv** — Team Member
- **Ganesha Thejaswi V** — Team Member

*Department of Computer Science & Engineering · MVJ College of Engineering*  
*From India, For India* 🇮🇳
