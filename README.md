# Drishti Kavach: AI-Powered Railway Obstacle and Track Clearance Perception System

**Drishti Kavach** is an indigenous computer vision perception and spatial clearance reasoning framework designed to augment Automatic Train Protection (ATP) systems, such as the Indian Railways' **Kavach**.

While conventional ATP platforms enforce speed governance and prevent Signal Passed at Danger (SPAD) or head-on/rear-end collisions through RFID transponders and trackside UHF radio telemetry, Drishti Kavach provides real-time forward optical track clearance monitoring, physical sabotage detection (iron rods, boulders), livestock/pedestrian intrusion analysis, and dynamic safety envelope reasoning across both daylight RGB and Active Infrared (850nm NIR) nocturnal CCTV streams.

---

## System Architecture

The Drishti Kavach pipeline comprises five core subsystems:

1. **Multi-Task Neural Network (`RailDrishti`)**:
   - Unified instance segmentation and object detection model (YOLO11-seg).
   - Simultaneously predicts continuous track bed polygons (`Rail_Track_Bed`), individual rail line paths (`Rail_Lines`), and 11-class foreign obstacle bounding boxes in a single forward pass.

2. **Adverse Weather Optical Enhancement Engine**:
   - Real-time atmospheric haze estimation.
   - Contrast Limited Adaptive Histogram Equalization (CLAHE) in the LAB luminance domain.
   - Koschmieder atmospheric scattering model inversion via Dark Channel Prior (DCP) defogging.
   - Temporal multi-frame median filtering for rain streak suppression.

3. **Spatial Hazard and Clearance Analyzer**:
   - 2D planar vector geometry computation using Shapely.
   - Evaluates obstacle ground-contact footprints against track geometry and dilated lateral safety envelopes.
   - Categorizes targets into:
     - **CRITICAL**: Obstacle directly within the track bed or rail lines (Emergency Braking).
     - **WARNING**: Obstacle within the lateral clearance envelope buffer (Caution).
     - **SAFE**: Obstacle located outside track clearance zones.

4. **High-Contrast Telemetry HUD Dashboard**:
   - Translucent Cyan-Blue track bed overlay and glowing emerald green rail lines.
   - Crisp threat bounding boxes with ground-contact anchor reticles.
   - Large, clean, old-school top status header with live FPS, sensor channel, and dynamic safety alerts.

5. **Universal Training & Validation Pipeline**:
   - Real-world readiness benchmark evaluator comparing training progress against actual Indian Railways operational deployment standards ($\ge 85\%$ mAP).

---

## Class Taxonomy (11 Classes)

| Class ID | Label | Category | Description |
| :--- | :--- | :--- | :--- |
| **0** | `Rail_Track_Bed` | Track Geometry | Ballast and sleeper envelope between outer track boundaries |
| **1** | `Rail_Lines` | Track Geometry | Structural steel running rails |
| **2** | `Branch` | Physical Obstacle | Fallen tree branches and foliage on track |
| **3** | `IronRod` | Sabotage / Debris | Slender metallic rods or track obstruction debris |
| **4** | `Barrel` | Physical Obstacle | Metal and plastic drums or containers |
| **5** | `Boulder` | Physical Obstacle | Rockfall and stone obstructions |
| **6** | `Jerrycan` | Physical Obstacle | Hazardous fuel containers and canisters |
| **7** | `Person` | Dynamic Threat | Pedestrians or trespassers on track right-of-way |
| **8** | `Cattle` | Dynamic Threat | Bovines (cows, bulls, buffaloes) |
| **9** | `Animal` | Dynamic Threat | Wildlife, canines, and livestock |
| **10** | `Vehicle` | Dynamic Threat | Road vehicles on level crossings or tracks |

---

## Repository Structure

```
drishti-kavach/
├── configs/
│   └── raildrishti_dataset.yaml      # Multi-task dataset configuration (Relative paths)
├── models/                           # Production directory for trained weights (RailDrishti.pt)
├── outputs/                          # Generated outputs, snapshots, HUD & day/night previews
│   ├── day_vs_night_previews/        # Side-by-side active IR CCTV conversion samples
│   ├── hud_previews/                 # Rendered Head-Up Display telemetry previews
│   ├── inference_results/            # Recorded inference video streams & images
│   └── snapshots/                    # High-resolution uncompressed image captures
├── src/
│   ├── augmentation/
│   │   └── night_cctv_converter.py   # Active IR (850nm NIR) physics-based sensor simulator
│   ├── colab_training/
│   │   ├── package_for_colab.py      # Dataset compression utility for Google Colab
│   │   └── train_on_colab.ipynb      # Cloud GPU (Tesla T4) training notebook with live monitor
│   ├── local_training/
│   │   └── train_local.py            # Universal local trainer (Mac MPS, Windows, Linux, Intel/AMD)
│   ├── pipeline/
│   │   ├── drishti_engine.py         # Real-time perception and reasoning coordinator
│   │   └── weather_enhancer.py       # Adaptive defogging, CLAHE, and rain filter engine
│   ├── preprocessing/
│   │   ├── unified_dataset_builder.py# 1080p mask parser and multi-task dataset generator
│   │   └── verify_annotations.py     # Ground-truth annotation visualizer
│   ├── spatial_reasoning/
│   │   └── hazard_analyzer.py        # Geometric polygon clearance reasoning engine
│   ├── visualization/
│   │   ├── generate_hud_preview.py   # HUD preview generator
│   │   └── visualizer.py             # High-contrast track overlay and telemetry HUD visualizer
│   └── camera_tester.py              # Hardware diagnostics and live camera snapshot tool
├── ALL_IN_ONE_GUIDE.md               # Full end-to-end technical guide and research methodology
├── run_inference.py                  # CLI runner for live webcams, videos, and RTSP streams
├── requirements.txt                  # Python package dependencies
└── .gitignore
```

---

## Setup and Installation

### 1. Environment Initialization

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Synthesis and Preprocessing

```bash
# Synthesize Active IR 850nm CCTV pairs across the raw dataset
python src/augmentation/night_cctv_converter.py --convert-all --workers 8

# Parse 1080p masks and generate unified multi-task dataset
python src/preprocessing/unified_dataset_builder.py

# Verify visual ground-truth parsing
python src/preprocessing/verify_annotations.py
```

---

## Model Training

### Option A: Local Laptop Training (Mac MPS, Windows, Linux)

Train directly on your machine with automatic hardware detection and the **Real-World Readiness Monitor**:

```bash
# Fast training (Recommended for laptops - 640px resolution):
python src/local_training/train_local.py --epochs 40 --batch 8 --imgsz 640

# High-precision full-resolution training (1024px):
python src/local_training/train_local.py --epochs 40 --batch 4 --imgsz 1024

# Resume an interrupted session:
python src/local_training/train_local.py --resume
```
*Trained weights are automatically saved to `models/RailDrishti.pt` upon completion.*

### Option B: Cloud GPU Training (Google Colab)

1. Compress the dataset:
   ```bash
   python src/colab_training/package_for_colab.py
   ```
2. Upload `raildrishti_colab.zip` to Google Drive root folder (`MyDrive`).
3. Open `src/colab_training/train_on_colab.ipynb` in Google Colab, select an **NVIDIA T4 GPU** runtime, and run all cells.
4. Place the downloaded `RailDrishti.pt` weight file into the local `models/` directory (`models/RailDrishti.pt`).

---

## Camera Hardware Diagnostics

Before running live tests, verify connected cameras and capture uncompressed test snapshots:

```bash
# 1. Scan and list all connected USB / UVC cameras:
python src/camera_tester.py --scan

# 2. Open live 1080p camera feed:
python src/camera_tester.py --cam 0

# Interactive Controls:
#   [S] / [SPACE] : Save snapshot to outputs/snapshots/
#   [C]           : Switch to next connected camera index
#   [F]           : Toggle Fullscreen
#   [Q] / [ESC]   : Quit
```

---

## Real-Time Inference and Deployment

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
- **`[D]`**: Cycle optical weather enhancement (`Auto` -> `CLAHE` -> `DCP Defog` -> `Rain Filter` -> `Off`).
- **`[H]`**: Toggle the HUD telemetry overlay on/off.
- **`[S]`**: Save high-resolution snapshot to `outputs/snapshots/`.
- **`[SPACE]`**: Pause or resume playback.
- **`[Q]` / `[ESC]`**: Terminate inference stream cleanly.

---

## Hardware Compatibility

- **Daylight Optical Streams**: Standard 1080p USB UVC webcams (e.g., Kreo Owl Lite FHD).
- **Night-Time Active IR Streams**: USB cameras equipped with motorized IR-Cut filters and onboard 850nm infrared LED illuminators (e.g., Arducam 1080P Day/Night USB Camera).
- **Station CCTV Infrastructure**: RTSP IP CCTV security cameras with EXIR 850nm illuminators.
