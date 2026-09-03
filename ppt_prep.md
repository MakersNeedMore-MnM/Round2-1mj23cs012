# Drishti-Kavach: PPT Preparation

---

## Slide 1: Abstract

Drishti-Kavach is an edge-deployable optical perception system designed to provide forward-looking track clearance verification for Indian Railways' Kavach (TCAS) network. It uses a decoupled dual-model vision pipeline combining BiSeNetV2 for real-time track and rail segmentation with YOLO11m for 8-class obstacle and sabotage detection. The system applies vector spatial geometry to classify threats into three distinct safety levels (Critical, Warning, Safe) and incorporates active 850nm Near-Infrared night vision along with adaptive defogging algorithms to maintain reliable, real-time operation at over 50 FPS in all weather conditions.

---

## Slide 2: Problem Statement

Modern train protection systems like Indian Railways' Kavach (TCAS) prevent train-to-train collisions using stationary trackside transponders and cab signaling, but they lack forward optical perception to detect physical track disruptions. Unforeseen hazards such as deliberate sabotage (placed steel rods and fishplates), fallen boulders, stalled vehicles at level crossings, and trespassers remain undetectable until within the driver's visual range. High locomotive speeds combined with long braking distances, poor night visibility, and severe weather like dense winter fog make manual visual detection inadequate, leading to preventable derailments and safety breaches.

---

## Slide 3: Proposed System

* **Decoupled Dual-Model Architecture:** Combines BiSeNetV2 for real-time drivable ballast and rail segmentation with a customized YOLO11m detector for 8 railway hazard classes.
* **Deterministic Vector Spatial Reasoning:** Uses polygon intersection geometry to evaluate ground-contact footprints against track boundaries, categorizing threats into Critical, Warning, and Safe zones.
* **24/7 Multi-Spectral Vision:** Employs active 850nm Near-Infrared processing paired with adaptive defogging (CLAHE and Dark Channel Prior) for round-the-clock reliability in heavy fog, rain, and darkness.
* **Automatic Train Protection Integration:** Provides direct telemetry signals to locomotive braking interfaces while rendering a real-time driver assistance HUD.
* **Edge-Deployable Performance:** Achieves over 50 FPS with an end-to-end processing latency under 20 ms on locomotive-grade hardware.

---

## Slide 4: Methodology

1. **Dataset Harmonization and Multi-Spectral Synthesis:** Standardized RailSem19 and UAV-RSOD datasets into unified 3-class segmentation and 8-class detection schemas, paired with synthetic 850nm Near-Infrared transformations.
2. **Atmospheric Optical Preprocessing:** Ingested video streams through dynamic fog density scoring, LAB-color CLAHE contrast enhancement, and Dark Channel Prior dehazing to restore visibility in severe weather.
3. **Decoupled Dual-Engine Inference:** Executed BiSeNetV2 at 512x1024 for ballast and rail line segmentation alongside YOLO11m at 1024x1024 for high-resolution obstacle and sabotage detection.
4. **Deterministic Vector Spatial Reasoning:** Converted segmentation masks into Shapely polygons, projected bottom 25% ground contact footprints for obstacles, and evaluated spatial intersections with dynamic clearance buffers.
5. **Safety Classification and Telemetry Output:** Assessed threat levels across Critical, Warning, and Safe categories to trigger automatic train protection braking signals and render real-time driver HUD overlays.

---

## Slide 5: System Specifications

### Hardware Components
* Forward-Facing Optical and 850nm NIR Night Vision Camera
* Active 850nm Infrared Spotlight Illuminator
* Embedded Edge AI Processing Unit (NVIDIA Jetson AGX Orin / Industrial GPU)
* In-Cab Driver Assistance Display (HUD)
* Locomotive Kavach (TCAS) ATP Braking Relay Interface

### Software Components
* Linux OS / Embedded RTOS Environment
* Python Runtime
* PyTorch Deep Learning Framework
* Ultralytics YOLO Library
* ONNX Runtime Engine
* OpenCV Computer Vision Library
* Shapely Vector Geometry Engine
* NumPy Scientific Computing Package

---

## Slide 6: System Design

* **Input Module:** Optical and 850nm Near-Infrared forward camera captures live 1080p track video feeds at 30 to 60 FPS, continuously delivering frames to the ingestion pipeline with minimal latency.
* **Communication Module:** Video frames are passed to the inference pipeline through thread-safe buffers, with real-time telemetry state records written periodically to synchronized data files and IPC sockets for dashboard and locomotive sub-rack integration.
* **Processing Module:** Performs atmospheric optical enhancement (CLAHE and Dark Channel Prior defogging) and runs parallel inference using BiSeNetV2 for track bed and rail segmentation alongside YOLO11m for 8-class hazard and sabotage detection.
* **Decision Module:** Vector spatial engine projects the bottom 25% ground-contact footprint of each detected object and evaluates geometric polygon intersections against track boundaries, categorizing events into Critical, Warning, and Safe states.
* **Output Module:** Renders a high-contrast driver assistance HUD with dynamic alert banners and distance overlays via OpenCV, while transmitting real-time telemetry signals to the locomotive Kavach Automatic Train Protection interface.

---

## Slide 7: Project Updates and Changelog

* **Raw Datasets Ingested:**
  * RailSem19: 8,500 locomotive cab-view frames across 38 countries with 19 railway classes.
  * UAV-RSOD (V1 and V2): Indian Railways drone and trackside infrastructure dataset for rail segmentation and 6 obstacle categories.

* **Curated Datasets Created:**
  * `dataset_segmentation`: Standardized 3-class cab segmentation (Background, Track Bed, Rail Lines) with 50/50 Day and 850nm NIR Night pairs.
  * `dataset_segmentation_uav_v1`: 3-class aerial/infrastructure segmentation for viewpoint regularization.
  * `dataset_detection`: Unified 8-class hazard and sabotage dataset with day/night pairs and strict train exclusion rules.

* **Hardware Camera Integration:**
  * Acquired 1080p Full HD camera with native 850nm Near-Infrared night vision sensor.
  * Features integrated active IR LED spotlight illumination, 30 to 60 FPS streaming, and low-latency USB capture.

* **Kaggle Cloud Training Cycle:**
  * Universal Dual Training: 75% Cab View + 25% Aerial View distribution with 50% Day + 50% NIR Night pairing over 50 epochs.
  * Trained BiSeNetV2 with AdamW and polynomial learning rate scheduling.
  * Fine-tuned YOLO11m at high resolution (1024x1024) using Task-Aligned Assigner on dual NVIDIA GPUs.

* **Final Production Models Generated:**
  * `RailDrishti_Seg_BiSeNetV2.onnx`: Optimized 13.3 MB edge model delivering 86.02% Universal mIoU and 90.08% ballast IoU.
  * `RailDrishti_Det_YOLO11m.onnx`: Optimized 80.8 MB detector delivering 60.80% mAP@50 overall and 90.90% AP@50 on deliberate sabotage (`IronRod`).
  * Base `yolo11m.pt`: Foundation model for complementary cattle, wildlife, and baggage detection.

---

## Slide 8: Environmental and Weather Algorithms

* **Automatic Fog Density Scorer:**
  * *Concept:* Automatically measures how much the image is washed out by fog.
  * *How it works:* Combines scene contrast variance with the brightness of dark regions (airlight intensity) into a 0.0 to 1.0 fog score to activate defogging only when necessary.

* **LAB-Domain Adaptive Contrast Enhancement (CLAHE):**
  * *Concept:* Restores visibility in moderate haze and low light without ruining original colors.
  * *How it works:* Converts the image to the LAB color space and applies contrast enhancement strictly to the Luminance (L) channel, sharpening ballast boundaries and rail lines while leaving color channels untouched.

* **Dark Channel Prior (DCP) Dehazing:**
  * *Concept:* Mathematically removes dense fog from the camera feed based on atmospheric physics.
  * *How it works:* Estimates how light scatters through moisture particles and calculates an inverted transmission map, recovering clear track details and distant obstacles hidden in heavy winter fog.

* **Temporal Rain-Streak Rolling Median Filter:**
  * *Concept:* Eliminates moving rain streaks from live video.
  * *How it works:* Compares pixel values across a 3-frame rolling buffer. Because falling rain drops move rapidly while railway tracks remain static, the median filter cancels out vertical rain streaks cleanly.

---

## Slide 9: Future Scope and Action Plan

* Add an auto-focus routine to the IR camera every 2 seconds to keep the tracks and distant objects sharp and clearly visible.
* Optimize the defogging and dehazing algorithms to run smoothly on low-power devices without dropping frame rates.
* Test the system outdoors during evening and night hours using physical IR filters to measure real-world night accuracy.

---

## Architecture Diagram Prompts for System Design Modules

### 1. Input Module Diagram Prompt
```text
Create a clean, professional architecture block diagram for the "Input Module" of a railway AI perception system called Drishti-Kavach.
Show the flow from left to right:
1. Physical Environment & Sensors: Dual-Spectrum Camera Unit (Daylight RGB Optical Sensor + 850nm Near-Infrared Night Vision Sensor) and Active 850nm IR Illuminator Spotlight.
2. Ingestion & Capture Interface: USB / MIPI CSI-2 Hardware Interface running at 1080p resolution (30 to 60 FPS).
3. Frame Conditioning Unit: Hardware-level frame synchronization, color space decoding (BGR format), and ring buffer allocation.
4. Output: Low-latency raw frame stream forwarded to the main perception pipeline.
Use standard engineering flowchart shapes, clean dark or neutral modern styling, clear labels, and arrows indicating real-time data flow.
```

### 2. Communication Module Diagram Prompt
```text
Create a clear, professional technical architecture diagram for the "Communication & Data Exchange Module" of Drishti-Kavach.
Show the internal components and data routing:
1. Frame Ingestion Thread: Thread-safe memory buffer receiving raw 1080p camera frames with drop-frame protection.
2. Inference Pipeline Inter-Process Communication (IPC): Shared memory queue passing processed tensors to the GPU worker threads.
3. Telemetry Dispatcher: Background publisher serializing track hazard states, FPS metrics, and threat levels into a structured JSON state cache every 500ms.
4. External Interfaces:
   - High-speed IPC / Socket connection to the In-Cab Driver HUD Display.
   - Hardware relay / CAN-bus / Serial interface to the Locomotive Kavach (TCAS) ATP sub-rack.
Use clean modular blocks with distinct color coding for input data, memory management, and external integration channels.
```

### 3. Processing Module Diagram Prompt
```text
Create a detailed, modern deep learning architecture diagram for the "Processing Module" of Drishti-Kavach.
Show the complete decoupled dual-engine pipeline:
1. Preprocessing Stage: Raw frame entering Atmospheric Optical Enhancer (Dark Channel Prior Fog Scorer, LAB-domain CLAHE, and 3-frame rolling median rain filter).
2. Parallel Model Split:
   - Upper Branch: BiSeNetV2 Semantic Segmenter (Input: 512x1024) containing Spatial Detail Branch (1/8 scale), Semantic Context Branch (1/32 scale + Context Embedding), and Bilateral Guided Aggregation (BGA) Layer. Output: 3-class segmentation mask (Background, Track Bed, Rail Lines).
   - Lower Branch: YOLO11m Obstacle Detector (Input: 1024x1024) with C3k2/C2PSA Backbone, Spatial Attention, and Decoupled Anchor-Free Detection Head. Output: 8-class bounding boxes for physical hazards and sabotage.
3. Feature Aggregation: OpenCV vector contour extraction converting pixel masks into track boundary coordinates.
Use professional technical diagram styling with highlighted tensor resolutions and clear parallel data pathways.
```

### 4. Decision Module Diagram Prompt
```text
Create a clear, mathematical architecture flow diagram for the "Vector Spatial Decision Module" of Drishti-Kavach.
Show the geometric reasoning pipeline:
1. Inputs: Segmented Track Bed & Rail Line Polygons from BiSeNetV2, and 2D Bounding Boxes from YOLO11m.
2. Ground Footprint Projection: Extraction of the bottom 25% base ground contact box and anchor base point for each detected object.
3. Track Vector Unification: Shapely Unary Union combining track bed and rail lines into a unified drivable polygon, plus a 65px dynamic lateral safety buffer zone.
4. Spatial Intersection & Clearance Logic:
   - Branch 1 (Footprint overlaps Track Geometry): Status = CRITICAL (In-Track Intrusion -> Assert Emergency Brake).
   - Branch 2 (Footprint within 65px Buffer): Status = WARNING (Near-Track Hazard -> Driver Caution Alert).
   - Branch 3 (Distance > 65px): Status = SAFE (Outside Clearance Envelope -> Normal Operation).
5. Output: Immutable HazardAssessment telemetry packet.
Use clean logic decision diamonds, color-coded status badges (Red, Amber, Green), and concise mathematical labels.
```

### 5. Output Module Diagram Prompt
```text
Create a sleek, high-level UI and telemetry architecture diagram for the "Output & Intervention Module" of Drishti-Kavach.
Show the dual output pathways from the Hazard Assessment decision:
1. Visual HUD Compositor (Driver Cockpit):
   - Real-time OpenCV rendering engine.
   - Translucent Cyan Track Bed Overlay (50% alpha) and Crimson Rail Contours.
   - Dynamic 3-Zone Telemetry Header (Hardware LED Lamp, Tactical Safety Banner, Live FPS, Sensor/Optics Mode).
   - Obstacle Ground Reticles and Threat Badges (Red / Amber / Green).
2. Hardware Automatic Train Protection (ATP) Interface:
   - Direct emergency braking relay signal to the Locomotive Kavach TCAS rack when status is CRITICAL.
   - Real-time diagnostic logs and telemetry broadcast for control room review.
Use modern presentation graphics style with clear visual separation between the driver display subsystem and the hardware brake actuation subsystem.
```

---
