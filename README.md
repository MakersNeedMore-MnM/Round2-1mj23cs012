# 🛡️ Drishti-Kavach (दृष्टि कवच)
### *Next-Generation Optical Automatic Train Protection (ATP) & Railway Clearance Perception Engine*

---

## 📌 Executive Summary
**Drishti-Kavach** is an advanced edge-deployable deep learning perception and geometric spatial reasoning system engineered for the **Indian Railways Kavach (TCAS - Train Collision Avoidance System)** network. 

Addressing the critical operational vulnerabilities of traditional track circuits and transponder-only collision prevention, Drishti-Kavach introduces an **autonomous forward-looking optical clearance envelope**. The system decouples deep multi-task perception into high-resolution semantic track bed segmentation and specialized physical obstacle detection, operating across diverse atmospheric conditions and dual-spectrum illumination (Daylight RGB and Active 850nm Near-Infrared Night Vision).

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DRISHTI-KAVACH PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  [ Forward Optical Feed ]  ──►  [ Atmospheric Optical Enhancer ] (Adaptive CLAHE / DCP)    │
│                                                   │                                         │
│                    ┌──────────────────────────────┴──────────────────────────────┐          │
│                    ▼                                                             ▼          │
│   [ BiSeNetV2 Semantic Segmenter ]                              [ YOLO11m Obstacle Detector ]│
│   • Track Bed Ballast (Drivable Gauge)                          • 8 Specialized Classes     │
│   • Steel Rail Ribbons (Distance Vector)                        • Sabotage & Hazard Items   │
│                    │                                                             │          │
│                    └──────────────────────────────┬──────────────────────────────┘          │
│                                                   ▼                                         │
│                         [ Spatial Hazard & Vector Clearance Engine ]                        │
│                         • Shapely Footprint Polygon Intersection                            │
│                         • Dynamic Lateral Safety Buffer Analysis                            │
│                         • 3-Tier Threat Classification (CRITICAL / WARNING / CLEAR)         │
│                                                   │                                         │
│                                                   ▼                                         │
│                         [ Real-Time Railway Telemetry HUD Dashboard ]                       │
│                         • Emergency Brake Trigger (ATP Interface)                           │
│                         • Driver HUD Distance & Clearance Telemetry                         │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 System Architecture & Core Modules

### 1. Decoupled Dual-Engine Perception
Unlike monolithic multi-task networks that compromise between small-object localization and dense boundary segmentation, Drishti-Kavach utilizes a decoupled dual-model architecture:

* **Semantic Track Segmentation Engine (`BiSeNetV2`)**:
  * Ultra-lightweight bilateral network architecture (~3.49M parameters) designed for real-time edge processing (>60 FPS).
  * **Detail Branch:** Preserves high-resolution spatial details and sharp rail ribbon boundaries at 1/8 scale.
  * **Semantic Branch:** Captures contextual field-of-view features with fast downsampling and global Context Embedding.
  * **Bilateral Guided Aggregation (BGA):** Fuses spatial detail cues with high-level semantic track guidance.
  * **Universal Dual Training:** Optimized on a joint distribution of locomotive cab viewpoints (RailSem19) and Indian Railways drone/ground infrastructure (UAV-RSOD V1) to eliminate catastrophic forgetting.

* **Obstacle & Sabotage Detection Engine (`YOLO11m`)**:
  * Real-time 2D bounding box detector operating at high resolution (1024x1024) for extended lookahead distances.
  * Specialized in detecting micro-obstructions, deliberate sabotage materials, road vehicles, and human trespassers before they enter braking distance limits.

---

### 2. Physical Hazard & Semantic Taxonomy

#### 🛤️ Semantic Segmentation Taxonomy (3 Classes)
| Class ID | Class Name | Representation | Function in Safety Envelope |
| :---: | :--- | :--- | :--- |
| **0** | `Background` | Non-rail environment, catenary, terrain, sky | Baseline non-drivable environment |
| **1** | `Track Bed` | Ballast drivable gauge footprint | Primary spatial clearance boundary |
| **2** | `Rail Lines` | Individual running steel rails | Horizon convergence & track center reference |

#### ⚠️ Physical Obstacle Detection Taxonomy (8 Classes)
| Class ID | Class Name | Target Description | Risk Level |
| :---: | :--- | :--- | :--- |
| **0** | `Person` | Pedestrians, trespassers, track gang maintenance workers | High |
| **1** | `Car` | Passenger motor vehicles stalled at level crossings | Critical |
| **2** | `Truck` | Commercial trucks, buses, tractors, heavy machinery | Critical |
| **3** | `Branch` | Fallen tree limbs, foliage debris from storm washouts | High |
| **4** | `IronRod` | Deliberate sabotage items (placed rails, steel rods, fishplates) | Critical |
| **5** | `Boulder` | Landslide rockfalls, placed ballast stones, debris | Critical |
| **6** | `Barrel` | Metal/plastic drums and containers on track | High |
| **7** | `Jerrycan` | Flammable fuel containers and hazardous canisters | High |

> **Strategic Filtering Design:** Trains, locomotives, rail wagons, and trams (`on-rails`) are intentionally excluded from obstacle detection to eliminate false alarms from parallel line traffic, sidings, and oncoming trains on adjacent tracks.

---

### 3. Spatial Hazard & Clearance Reasoning (ATP Logic)
The **Spatial Hazard Analyzer** translates 2D object detections and segmented track masks into mathematical vector polygons using Shapely vector geometry:

* **Footprint Projection:** Calculates the bottom-edge ground contact polygon of each detected obstacle.
* **Three-Tier Threat Classification:**
  1. 🔴 **CRITICAL (In-Track):** Obstacle footprint directly overlaps with the segmented **Track Bed** or **Rail Lines**. Immediately asserts emergency braking telemetry to the locomotive ATP interface.
  2. 🟡 **WARNING (Near-Track):** Obstacle is located outside the active drivable gauge but falls within the dynamic lateral clearance safety envelope (e.g., within 65 pixels lateral buffer). Alerts driver HUD for heightened vigilance.
  3. 🟢 **CLEAR / SAFE (Off-Track):** Obstacle is situated completely outside the railway clearance envelope (e.g., pedestrian on distant platform or vehicle on parallel roadway). No brake intervention required.

---

### 4. Dual-Spectrum Sensor & Environmental Robustness
* **Daylight RGB Imaging:** Natural high-contrast sunlight illumination across urban, rural, and mountainous terrains.
* **Active 850nm Near-Infrared (NIR) Night Vision:** Physics-accurate active infrared railway headlamp modeling with radial intensity decay, localized spotlight beam cones, and atmospheric backscatter mitigation.
* **Atmospheric Optical Enhancer:** Multi-mode adaptive defogging integrating Contrast Limited Adaptive Histogram Equalization (CLAHE), Dark Channel Prior (DCP) atmospheric transmission estimation, and dynamic fog-density scoring.

---

## 🗂️ Project Directory Structure

```
drishti-kavach/
│
├── docs/                                    # Technical guides, taxonomies & benchmark protocols
│   ├── models_accuracy_metrics_guide.txt    # Deep learning architectures & accuracy metrics guide
│   └── obstacle_taxonomy.txt                # Unified dual-layer obstacle & sabotage taxonomy
│
├── models/                                  # Trained checkpoints & production ONNX models
│   ├── RailDrishti_Seg_BiSeNetV2.pth        # Universal Dual Track Segmenter Checkpoint (86.02% mIoU)
│   ├── RailDrishti_Seg_BiSeNetV2.onnx       # Universal Dual Track Segmenter ONNX Export (13.3 MB)
│   ├── RailDrishti_Det_YOLO11m.pt           # Custom 8-Class Railway Sabotage Detector (60.8% mAP50)
│   ├── RailDrishti_Det_YOLO11m.onnx         # Custom 8-Class Railway Sabotage Detector ONNX Export
│   └── yolo11m.pt                           # Base Foundation YOLO11m Model
│
├── research_paper/                          # IEEE publication manuscript, figures & literature
│   ├── figures/                             # High-resolution architectural figures & HUD captures
│   ├── literature_papers/                   # Foundational reference papers & state-of-the-art literature
│   ├── IEEE_RESEARCH_PAPER.md               # Markdown version of full IEEE research paper
│   ├── IEEE_RESEARCH_PAPER.pdf              # Compiled IEEE research paper PDF
│   ├── IEEEtran.cls                         # Official IEEE LaTeX document class
│   ├── main.tex                             # LaTeX manuscript source code
│   └── main.pdf                             # Typeset publication-ready manuscript PDF
│
├── src/                                     # Core source code modules
│   │
│   ├── 1_preprocessing/                     # Dataset builders & dual-spectrum pipelines
│   │   ├── prep_railsem19_segmentation.py   # Global locomotive cab segmentation builder
│   │   ├── prep_uav_v1_segmentation.py      # Indian Railways track & aerial mask builder
│   │   └── prep_obstacle_detection.py       # 8-Class physical obstacle dataset builder
│   │
│   ├── 2_models/                            # Deep learning model architectures
│   │   └── bisenetv2.py                     # BiSeNetV2 architecture (~3.49M params)
│   │
│   ├── 3_kaggle_train/                      # Cloud distributed training workflows
│   │   ├── package_for_kaggle.py            # RailSem19 dataset packager
│   │   ├── package_uav_v1_kaggle.py         # UAV-RSOD V1 dataset packager
│   │   ├── package_detection_kaggle.py      # Obstacle detection dataset packager
│   │   ├── train_bisenetv2_kaggle.ipynb     # Base BiSeNetV2 trainer notebook
│   │   ├── train_universal_bisenetv2_kaggle.ipynb # Universal Dual trainer notebook
│   │   └── train_yolo11m_kaggle.ipynb       # YOLO11m 8-class obstacle detector notebook
│   │
│   ├── 4_local_train/                       # Local training & validation suites
│   │   ├── train_local.py                   # Local BiSeNetV2 training script (MPS/CUDA)
│   │   └── train_yolo11m_local.py           # Local YOLO11m training script (MPS/CUDA)
│   │
│   ├── 5_spatial_reasoning/                 # Clearance geometry & ATP decision engine
│   │   └── hazard_analyzer.py               # Vector Shapely clearance reasoning
│   │
│   ├── 6_visualization/                     # Telemetry rendering & HUD overlays
│   │   └── visualizer.py                    # Railway HUD & segmented mask compositor
│   │
│   └── 7_pipeline/                          # End-to-end perception orchestration
│       ├── drishti_engine.py                # Decoupled Dual-Engine master pipeline
│       └── weather_enhancer.py              # Atmospheric defogger & CLAHE optimizer
│
├── test_samples/                            # Evaluation & verification test samples
│   ├── sample_images/                       # Daylight railway test frames
│   ├── sample_images_night/                 # Active 850nm NIR night vision test frames
│   └── sample_videos/                       # Full-motion locomotive video feed
│
├── accuracy_metrics_test.py                 # Comprehensive accuracy & benchmark test suite
├── camera_tester.py                         # Cross-platform live camera stream & snapshot tool
├── FULL_DOCUMENTATION.md                    # Exhaustive 10-phase engineering & architecture documentation
├── README.md                                # Project overview & quick start guide
├── requirements.txt                         # Python dependencies manifest
└── main.py                                  # Primary real-time CLI & GUI inference engine
```

---

## ⚡ Quick Start & Usage

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/alvinxsonny/drishti-kavach.git
cd drishti-kavach

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Real-Time Inference
```bash
# Run inference on live camera feed (Device 0)
python main.py --source 0

# Run inference on a sample image and save results
python main.py --source test_samples/sample_images/1.jpg --save

# Run inference with Active 850nm NIR Night Vision sensor preset
python main.py --source test_samples/sample_images_night/1.jpg --sensor "850nm ACTIVE IR CCTV" --save

# Run on a video file in headless mode
python main.py --source test_samples/sample_videos/test.mp4 --no-view --save
```

### 3. Model Accuracy & Benchmark Suite
```bash
# Fast evaluation benchmark
python accuracy_metrics_test.py

# Benchmark with custom sample count
python accuracy_metrics_test.py --samples 100

# Full comprehensive evaluation across all test sets
python accuracy_metrics_test.py --full
```

### 4. Camera Diagnostics Tool
```bash
# Probe connected cameras and launch zero-latency preview
python camera_tester.py
```

---

## 📊 Benchmark Metrics Summary

| Perception Subsystem | Architecture / Model | Input Resolution | Parameter Count | Key Evaluation Metric | Performance Target |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Track Bed Segmentation** | BiSeNetV2 (Universal) | 512x1024 | ~3.49M | Track Bed IoU | **>89.5%** |
| **Rail Line Segmentation** | BiSeNetV2 (Universal) | 512x1024 | ~3.49M | Rail Lines IoU | **>68.5%** |
| **Overall Segmentation** | BiSeNetV2 (Universal) | 512x1024 | ~3.49M | Mean IoU (mIoU) | **>85.5%** |
| **Obstacle Detection** | YOLO11m | 1024x1024 | ~20.1M | mAP@50 | **>91.0%** |
| **Inference Latency** | Decoupled Pipeline | Combined | Total Engine | End-to-End Speed | **>45 FPS (Edge)** |

---

## 📖 Extended Documentation
* For exhaustive technical details, data preparation steps, model training logs, mathematical formulations, and engineering architecture, refer to [FULL_DOCUMENTATION.md](file:///Users/alvinsonny/Desktop/drishti-kavach/FULL_DOCUMENTATION.md).
* For detailed guidance on model weights, metrics calculations, and loss formulas, see [docs/models_accuracy_metrics_guide.txt](file:///Users/alvinsonny/Desktop/drishti-kavach/docs/models_accuracy_metrics_guide.txt).
* For the complete 8-class obstacle and 19-class complementary foundation taxonomy, see [docs/obstacle_taxonomy.txt](file:///Users/alvinsonny/Desktop/drishti-kavach/docs/obstacle_taxonomy.txt).
* For the academic publication manuscript and LaTeX source, see [research_paper/](file:///Users/alvinsonny/Desktop/drishti-kavach/research_paper/).
