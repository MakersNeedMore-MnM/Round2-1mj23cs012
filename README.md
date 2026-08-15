# Drishti Kavach: AI-Powered Railway Obstacle and Track Clearance Perception System

Drishti Kavach is a computer vision perception and spatial clearance reasoning framework designed to augment Automatic Train Protection (ATP) systems, such as the Indian Railways' Kavach. 

While conventional ATP platforms prevent signal passing at danger (SPAD) and head-on/rear-end collisions through RFID and trackside radio telemetry, Drishti Kavach provides real-time optical track clearance monitoring, foreign object detection, and dynamic clearance envelope breach analysis across both daylight RGB and active infrared (850nm NIR) night surveillance feeds.

---

## System Architecture

The Drishti Kavach pipeline comprises four core functional subsystems:

1. **Multi-Task Neural Network (RailDrishti)**:
   - Unified instance segmentation and object detection model (YOLO11-seg).
   - Simultaneously predicts continuous track bed polygons (`Rail_Track_Bed`), individual rail line paths (`Rail_Lines`), and multi-class foreign obstacle bounding boxes in a single forward pass.

2. **Optical Weather Enhancement Engine**:
   - Real-time atmospheric haze index estimation.
   - Contrast Limited Adaptive Histogram Equalization (CLAHE) in the LAB luminance domain.
   - Atmospheric Scattering Model inversion via Dark Channel Prior (DCP) defogging.
   - Temporal multi-frame median filtering for rain streak suppression.

3. **Spatial Hazard and Clearance Analyzer**:
   - Vector geometry computation using polygon intersections.
   - Evaluates obstacle ground-contact footprints against track geometry and lateral safety envelopes.
   - Categorizes targets into:
     - **CRITICAL**: Obstacle directly within the track bed or rail lines (Collision Threat).
     - **WARNING**: Obstacle within the lateral clearance envelope buffer.
     - **SAFE**: Obstacle located outside track clearance zones.

4. **High-Contrast Telemetry HUD Dashboard**:
   - Translucent Cyan-Blue track bed overlay and glowing emerald green rail lines.
   - Color-coded threat bounding boxes with dynamic confidence and distance badges.
   - Real-time telemetry bar indicating system state, frame rate, sensor channel, and active hazards.

---

## Class Taxonomy

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
│   └── raildrishti_dataset.yaml      # Multi-task dataset configuration
├── models/                           # Target directory for trained weights (RailDrishti.pt)
├── outputs/                          # Generated outputs, snapshots, and verification samples
├── src/
│   ├── augmentation/
│   │   └── night_cctv_converter.py   # Active IR (850nm NIR) physics-based sensor simulator
│   ├── colab_training/
│   │   ├── package_for_colab.py      # Dataset compression utility for Google Colab
│   │   └── train_on_colab.ipynb      # Cloud GPU (Tesla T4) training notebook
│   ├── pipeline/
│   │   ├── drishti_engine.py         # Real-time perception and reasoning coordinator
│   │   └── weather_enhancer.py       # Adaptive defogging, CLAHE, and rain filter engine
│   ├── preprocessing/
│   │   ├── unified_dataset_builder.py# 1080p mask parser and multi-task dataset generator
│   │   └── verify_annotations.py     # Ground-truth annotation visualizer
│   ├── spatial_reasoning/
│   │   └── hazard_analyzer.py        # Geometric polygon clearance reasoning engine
│   └── visualization/
│       └── visualizer.py             # High-contrast track overlay and telemetry HUD visualizer
├── run_inference.py                  # CLI runner for webcam, video, and image streams
├── requirements.txt                  # Python package dependencies
└── .gitignore
```

---

## Setup and Installation

### 1. Environment Initialization

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

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

### 3. Model Training (Google Colab GPU)

1. Compress the dataset:
   ```bash
   python src/colab_training/package_for_colab.py
   ```
2. Upload `raildrishti_colab.zip` to Google Drive.
3. Open `src/colab_training/train_on_colab.ipynb` in Google Colab, select a T4 GPU runtime, and run all cells.
4. Place the downloaded `RailDrishti.pt` checkpoint into the local `models/` directory (`models/RailDrishti.pt`).

---

## Real-Time Inference and Deployment

### 1. Live USB Webcam Stream (Daylight Station Testing)
```bash
python run_inference.py --source 0 --weather auto
```

### 2. Night Vision Mode (850nm Active IR USB Camera)
```bash
python run_inference.py --source 0 --sensor "850nm ACTIVE IR CCTV"
```

### 3. Processing Video Files with Output Recording
```bash
python run_inference.py --source path/to/rail_video.mp4 --weather auto --save
```

### 4. Interactive HUD Keyboard Controls

During active video playback:
- **`[D]`**: Cycle weather enhancement modes (`Auto` -> `CLAHE` -> `DCP Defog` -> `Rain Filter` -> `Off`).
- **`[H]`**: Toggle the HUD telemetry overlay.
- **`[S]`**: Save high-resolution snapshot to `outputs/snapshots/`.
- **`[SPACE]`**: Pause or resume playback.
- **`[Q]` / `[ESC]`**: Terminate inference stream.

---

## Hardware Compatibility

- **Daylight Perception**: Standard 1080p USB UVC webcams (e.g., Kreo Owl Lite FHD).
- **Night-Time Perception**: USB cameras equipped with automatic motorized IR-Cut filters and onboard 850nm infrared LED illuminators (e.g., Arducam 1080P Day/Night USB Camera, SKU: `R114160` / `B0506`).
- **Station Infrastructure**: RTSP IP CCTV security cameras with EXIR 850nm arrays.
