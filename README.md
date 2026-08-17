# 🛡️ Drishti Kavach: Indigenous AI-Powered Railway Obstacle & Track Clearance Perception System

**Drishti Kavach** is an indigenous, real-time computer vision perception and spatial clearance reasoning framework designed to augment Automatic Train Protection (ATP) systems, such as the Indian Railways' **Kavach**.

While conventional ATP platforms enforce speed governance and prevent Signal Passed at Danger (SPAD) or head-on/rear-end collisions through RFID track transponders and stationary UHF radio telemetry, **Drishti Kavach** provides real-time forward optical track clearance monitoring, physical sabotage detection (e.g., placed iron rods, boulders, debris), livestock and pedestrian intrusion alerts, and dynamic clearance envelope reasoning across both daylight RGB and Active Infrared (850nm NIR) nocturnal CCTV streams.

---

## 🏛️ System Architecture

The Drishti Kavach pipeline comprises five core integrated subsystems:

```
                  ┌──────────────────────────────────────────────────┐
                  │ 📷 SENSOR STREAM (Daylight RGB / 850nm NIR CCTV) │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │ 🌦️ OPTICAL WEATHER ENHANCEMENT ENGINE            │
                  │ (Dark Channel Prior Defogging + LAB CLAHE)       │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │ 🧠 MULTI-TASK NEURAL NETWORK (RailDrishti)       │
                  │ • Track Bed & Rail Lines Instance Segmentation   │
                  │ • 11-Class Foreign Hazard Bounding Box Detection │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │ 📐 2D GEOMETRIC CLEARANCE & HAZARD ANALYZER     │
                  │ (CRITICAL: In-Gauge | WARNING: Buffer | SAFE)    │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │ 📟 TACTICAL TELEMETRY HUD & KAVACH ATP ALERT     │
                  │ (Cyan-Blue Track Bed, Maroon Rails, Live Pilot)  │
                  └──────────────────────────────────────────────────┘
```

1. **Optical Weather Enhancement Engine**: Real-time atmospheric haze estimation, Contrast Limited Adaptive Histogram Equalization (CLAHE) in the LAB luminance domain, Koschmieder scattering model inversion via Dark Channel Prior (DCP) defogging, and rain streak suppression.
2. **Multi-Task Neural Network (`RailDrishti`)**: Unified instance segmentation and object detection model (YOLO11-seg) predicting continuous track bed polygons (`Rail_Track_Bed`), individual running rail paths (`Rail_Lines`), and 11-class hazard bounding boxes in a single forward pass.
3. **Spatial Hazard and Clearance Analyzer**: 2D planar vector geometry computation using Shapely. Evaluates obstacle ground-contact footprints against track geometry and dilated lateral safety buffers (Categorizes into **`CRITICAL`**, **`WARNING`**, and **`SAFE`**).
4. **High-Contrast Telemetry HUD Dashboard**: Translucent Cyan-Blue track bed overlay, solid maroon running rails, red threat bounding boxes, and an 80px old-school tactical top status header with live FPS, sensor mode, and Kavach emergency braking alerts.
5. **Universal Training & Validation Engine**: Real-world readiness benchmark evaluator comparing training progress against actual Indian Railways operational deployment standards ($\ge 85\%$ mAP).

---

## 🏷️ Class Taxonomy (11 Classes)

| Class ID | Label | Type | Category | Description |
| :---: | :--- | :---: | :--- | :--- |
| **0** | `Rail_Track_Bed` | Polygon | Track Geometry | Ballast and sleeper envelope between outer track boundaries |
| **1** | `Rail_Lines` | Polygon | Track Geometry | Structural steel running rails (left and right rails) |
| **2** | `Branch` | BBox | Physical Obstacle | Fallen tree branches and foliage on track |
| **3** | `IronRod` | BBox | Sabotage / Debris | Slender metallic rods or intentional track obstruction debris |
| **4** | `Barrel` | BBox | Physical Obstacle | Metal and plastic drums or containers |
| **5** | `Boulder` | BBox | Physical Obstacle | Rockfall and stone obstructions |
| **6** | `Jerrycan` | BBox | Hazardous Flammable | Fuel containers and canisters |
| **7** | `Person` | BBox | Dynamic Threat | Pedestrians or trespassers on track right-of-way |
| **8** | `Cattle` | BBox | Dynamic Threat | Bovines (cows, bulls, buffaloes) |
| **9** | `Animal` | BBox | Dynamic Threat | Wildlife, canines, and livestock |
| **10** | `Vehicle` | BBox | Dynamic Threat | Road vehicles on level crossings or tracks *(Excludes parallel trains)* |

---

## 📁 Repository Structure

```
drishti-kavach/
├── configs/
│   └── raildrishti_dataset.yaml      # Unified multi-task dataset configuration
├── models/                           # Production directory for trained weights (RailDrishti.pt)
├── outputs/                          # Generated outputs, snapshots, and verification previews
│   ├── day_vs_night_previews/        # Side-by-side Active IR CCTV conversion samples
│   ├── hud_previews/                 # Rendered Head-Up Display telemetry previews
│   ├── inference_results/            # Recorded inference video streams & images
│   ├── snapshots/                    # High-resolution uncompressed image captures
│   └── verification_samples/         # 20-sample structured ground-truth verification renders
├── src/
│   ├── augmentation/
│   │   └── night_cctv_converter.py   # Active IR (850nm NIR) physics-based sensor simulator
│   ├── colab_training/
│   │   ├── package_for_colab.py      # High-speed zip packager for Google Colab GPU training
│   │   └── train_on_colab.ipynb      # Cloud GPU (Tesla T4/A100) training notebook with live monitor
│   ├── local_training/
│   │   └── train_local.py            # Universal local trainer with Pause/Resume & Color-Coded Monitor
│   ├── pipeline/
│   │   ├── drishti_engine.py         # Real-time perception and reasoning coordinator
│   │   └── weather_enhancer.py       # Adaptive defogging, CLAHE, and rain filter engine
│   ├── preprocessing/
│   │   ├── unified_dataset_builder.py# Curated hybrid builder for RailSem19 + UAV-RSOD V1 + V2
│   │   └── verify_annotations.py     # 20-sample structured visual ground-truth validator
│   ├── spatial_reasoning/
│   │   └── hazard_analyzer.py        # Geometric polygon clearance reasoning engine
│   ├── visualization/
│   │   ├── generate_hud_preview.py   # HUD preview generator
│   │   └── visualizer.py             # High-contrast track overlay and telemetry HUD visualizer
│   └── camera_tester.py              # Hardware diagnostics and live camera snapshot tool
├── dataset_guide.md                  # Comprehensive dataset architecture & curation guide
├── ALL_IN_ONE_GUIDE.md               # End-to-end engineering methodology and research guide
├── run_inference.py                  # CLI runner for live webcams, videos, and RTSP streams
├── requirements.txt                  # Python package dependencies
└── .gitignore
```

---

## ⚙️ Setup and Installation

### 1. Environment Initialization

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚆 Dataset Construction Pipeline

Drishti Kavach utilizes a curated hybrid dataset of **~12,900 Full-HD (1080p) Images** combining:
1. **RailSem19**: Train driver cab POV scenes (track beds, running rails, switches, crossings, pedestrians, road vehicles).
2. **UAV-RSOD V1**: Dense 1080p ground-truth masks for `Rail_Track_Bed` and `Rail_Lines`.
3. **UAV-RSOD V2**: Specialized physical sabotage hazards (`IronRod`, `Boulder`, `Jerrycan`, `Barrel`, `Branch`, `Cattle`).
4. **Active IR (850nm NIR)**: Matched 1:1 nocturnal security CCTV counterparts.

### Step 1: Build the Curated Balanced Daylight Dataset
```bash
python src/preprocessing/unified_dataset_builder.py
```
*(Extracts ~6,450 curated daylight images into `dataset_rail-drishti/` and generates `configs/raildrishti_dataset.yaml`)*.

### Step 2: Synthesize Active IR (850nm) Nocturnal Pairs
```bash
python src/augmentation/night_cctv_converter.py --convert-all --workers 8
```
*(Synthesizes matched 850nm night-vision images directly inside `dataset_rail-drishti/`, bringing the total dataset to **12,902 images**)*.

### Step 3: Visually Verify Annotations (20 Structured Previews)
```bash
python src/preprocessing/verify_annotations.py
```
*(Generates 5 previews each for RailSem19 Track Seg, RailSem19 Obstacles, UAV-RSOD V1 Seg, and UAV-RSOD V2 Obstacles in `outputs/verification_samples/`)*.

---

## 🧠 Model Training

### Option A: Cloud GPU Training (Google Colab — Recommended for 1024px Full Res)

1. **Package the dataset**:
   ```bash
   python src/colab_training/package_for_colab.py
   ```
2. Upload the generated `raildrishti_colab.zip` to your **Google Drive root folder (`MyDrive`)**.
3. Open [`src/colab_training/train_on_colab.ipynb`](src/colab_training/train_on_colab.ipynb) in **Google Colab**.
4. Set **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**, and click **Run All**.
   *(When finished, it automatically saves `RailDrishti.pt` to Google Drive and downloads it to your computer)*.

---

### Option B: Local Training (MacBook MPS GPU, Windows CUDA, or Linux)

Train locally with automatic hardware acceleration, live red-to-green accuracy monitoring, and **graceful pause/resume**:

```bash
# 1. Fast training (Recommended for laptops - 640px resolution):
python src/local_training/train_local.py --epochs 40 --batch 8 --imgsz 640

# 2. High-precision full-resolution training (1024px):
python src/local_training/train_local.py --epochs 40 --batch 4 --imgsz 1024

# 3. Pause & Resume Control:
#    • Press [Ctrl + C] anytime to pause safely without losing progress.
#    • Resume anytime with:
python src/local_training/train_local.py --resume
```
*Trained weights are automatically saved to `models/RailDrishti.pt` upon completion.*

---

## 📷 Camera Hardware Diagnostics

Verify connected USB/UVC cameras and capture test snapshots:

```bash
# 1. Scan and list all connected USB / UVC cameras:
python src/camera_tester.py --scan

# 2. Open live 1080p camera feed:
python src/camera_tester.py --cam 0

# Interactive Controls:
#   [S] / [SPACE] : Save uncompressed snapshot to outputs/snapshots/
#   [C]           : Switch to next connected camera index
#   [F]           : Toggle Fullscreen
#   [Q] / [ESC]   : Quit
```

---

## 🚦 Real-Time Inference & Station Deployment

### 1. Live USB Webcam Stream (Station & Field Testing)
```bash
python run_inference.py --source 0 --weather auto
```

### 2. Night Vision Mode (850nm Active IR USB Camera)
```bash
python run_inference.py --source 0 --sensor "850nm ACTIVE IR CCTV"
```

### 3. Processing Video Files with Output Recording
```bash
python run_inference.py --source data/rail_video.mp4 --weather auto --save
```

### 4. Interactive HUD Keyboard Controls
During active video playback:
- **`[D]`**: Cycle optical weather enhancement (`Auto` $\rightarrow$ `CLAHE` $\rightarrow$ `DCP Defog` $\rightarrow$ `Rain Filter` $\rightarrow$ `Off`).
- **`[H]`**: Toggle the HUD telemetry overlay on/off.
- **`[S]`**: Save high-resolution snapshot to `outputs/snapshots/`.
- **`[SPACE]`**: Pause or resume playback.
- **`[Q]` / `[ESC]`**: Terminate inference stream cleanly.

---

## 🔌 Hardware Compatibility

- **Daylight Perception**: Standard 1080p USB UVC webcams (e.g., Kreo Owl Lite FHD).
- **Night-Time Perception**: USB cameras equipped with motorized IR-Cut filters and onboard 850nm infrared LED illuminators (e.g., **Arducam 1080P Day/Night USB Camera B0205** / ELP 1080P Day/Night Camera).
- **Station Infrastructure**: RTSP IP CCTV security cameras with EXIR 850nm illuminators.
