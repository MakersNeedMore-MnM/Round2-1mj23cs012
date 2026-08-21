"""
Drishti Kavach: End-to-End Real-Time Perception & Decision Pipeline (Decoupled Dual-Engine)

Integrates:
  1. Weather & Atmospheric Optical Enhancer (Adaptive Defogger / CLAHE)
  2. BiSeNetV2 Semantic Track & Rail Segmentation Engine (3 Classes)
  3. YOLO11m Railway Obstacle & Sabotage Detection Engine (8 Classes)
  4. Spatial Hazard & Clearance Envelope Reasoning Engine (Shapely Vector Geometry)
  5. Real-Time Railway Telemetry HUD Dashboard Renderer
"""

import os
import time
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
import cv2
import numpy as np
import torch
import importlib
from ultralytics import YOLO

_models = importlib.import_module("src.2_models.bisenetv2")
BiSeNetV2 = _models.BiSeNetV2

_weather = importlib.import_module("src.7_pipeline.weather_enhancer")
WeatherEnhancer = _weather.WeatherEnhancer

_spatial = importlib.import_module("src.5_spatial_reasoning.hazard_analyzer")
SpatialHazardAnalyzer = _spatial.SpatialHazardAnalyzer
HazardAssessment = _spatial.HazardAssessment

_vis = importlib.import_module("src.6_visualization.visualizer")
DrishtiVisualizer = _vis.DrishtiVisualizer


OBSTACLE_CLASSES = {
    0: "Person",
    1: "Car",
    2: "Truck",
    3: "Branch",
    4: "IronRod",
    5: "Boulder",
    6: "Barrel",
    7: "Jerrycan"
}


# Selected complementary COCO foundation classes for Indian Railways
BASE_COMPLEMENTARY_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    13: "Bench",
    15: "Cat",
    16: "Dog",
    17: "Horse",
    19: "Cow",
    20: "Elephant",
    21: "Bear",
    22: "Zebra",
    23: "Giraffe",
    24: "Backpack",
    25: "Umbrella",
    26: "Handbag",
    28: "Suitcase"
}


class DrishtiEngine:
    """
    Complete real-time perception pipeline for Indian Railways Optical Kavach.
    Integrates Universal Track Segmentation + Dual-Layer Obstacle & Sabotage Detection.
    """

    def __init__(
        self,
        seg_model_path: str = "models/RailDrishti_Seg_Universal.pth",
        det_model_path: str = "models/RailDrishti_Det_YOLO11m.pt",
        base_det_model_path: str = "yolo11m.pt",
        conf_thresh: float = 0.35,
        imgsz: int = 1024,
        device: Optional[str] = None,
        weather_mode: str = "auto",
        warning_buffer_px: float = 65.0,
        enable_base_detector: bool = True
    ):
        self.seg_model_path = seg_model_path
        self.det_model_path = det_model_path
        self.base_det_model_path = base_det_model_path
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.weather_mode = weather_mode
        self.enable_base_detector = enable_base_detector

        # Hardware selection
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
                self.torch_device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = "0"
                self.torch_device = torch.device("cuda")
            else:
                self.device = "cpu"
                self.torch_device = torch.device("cpu")
        else:
            self.device = device
            self.torch_device = torch.device(device if device != "0" else "cuda")

        print("=" * 75)
        print(" 🛡️  INITIALIZING DRISHTI KAVACH REAL-TIME ENGINE (DUAL-LAYER)")
        print("=" * 75)
        print(f" • Segmentation Model: {self.seg_model_path}")
        print(f" • Custom Det Model:   {self.det_model_path}")
        print(f" • Base Det Model:     {self.base_det_model_path} ({len(BASE_COMPLEMENTARY_CLASSES)} classes enabled)")
        print(f" • Hardware Device:    {self.device}")
        print(f" • Target Resolution:  {imgsz}x{imgsz}")
        print(f" • Confidence Cutoff:  {conf_thresh:.2f}")
        print(f" • Weather Optimizer:  {weather_mode.upper()}")
        print("=" * 75)

        # 1. Load BiSeNetV2 Semantic Segmentation Model
        self.seg_model = BiSeNetV2(num_classes=3, is_training=False).to(self.torch_device)
        
        actual_seg_path = self.seg_model_path
        if not os.path.exists(actual_seg_path):
            fallback_path = "models/best_bisenetv2_raildrishti.pth"
            if os.path.exists(fallback_path):
                actual_seg_path = fallback_path

        if os.path.exists(actual_seg_path):
            state_dict = torch.load(actual_seg_path, map_location=self.torch_device)
            infer_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux")}
            self.seg_model.load_state_dict(infer_state_dict, strict=False)
            print(f"[+] Loaded BiSeNetV2 Track Model from: {actual_seg_path}")
        else:
            print(f"[!] Warning: No segmentation checkpoint found at {actual_seg_path}.")

        self.seg_model.eval()

        # 2. Load Custom YOLO11m Obstacle Detector (Layer 1: Sabotage & Railway Threats)
        if os.path.exists(self.det_model_path):
            self.custom_det = YOLO(self.det_model_path)
            print(f"[+] Loaded Custom Railway Detector from: {self.det_model_path}")
        else:
            print(f"[*] Custom weights '{self.det_model_path}' not found yet. Using 'yolo11m.pt'.")
            self.custom_det = YOLO("yolo11m.pt")

        self.custom_names = OBSTACLE_CLASSES

        # 3. Load Base Foundation YOLO11m Model (Layer 2: Animals, Luggage & General Transport)
        self.base_det = None
        if self.enable_base_detector:
            try:
                self.base_det = YOLO(self.base_det_model_path)
                print(f"[+] Loaded Complementary Base Foundation Detector: {self.base_det_model_path}")
            except Exception as e:
                print(f"[!] Warning: Could not load base foundation detector: {e}")

        # Subsystems
        self.weather_enhancer = WeatherEnhancer()
        self.hazard_analyzer = SpatialHazardAnalyzer(warning_buffer_pixels=warning_buffer_px)
        self.visualizer = DrishtiVisualizer()

        # Image normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # FPS metrics
        self.prev_time = time.time()
        self.fps = 0.0

    def _extract_polygons_from_mask(self, mask: np.ndarray, min_area: int = 50) -> List[np.ndarray]:
        """Extracts boundary polygons from a binary mask."""
        polys = []
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                pts = cnt.squeeze(1)
                if len(pts.shape) == 2 and pts.shape[0] >= 3:
                    polys.append(pts)
        return polys

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        sensor_type: str = "DAYLIGHT RGB",
        show_hud: bool = True,
        is_video_stream: bool = False
    ) -> Tuple[np.ndarray, str, List[HazardAssessment], Dict]:
        """
        Executes full perception and reasoning cycle on a single video frame.
        """
        t_start = time.time()
        h_orig, w_orig = frame_bgr.shape[:2]

        # 1. Optical Enhancement & Defogging
        enhanced_frame, weather_status, fog_score = self.weather_enhancer.process(
            frame_bgr, mode=self.weather_mode, is_video_stream=is_video_stream
        )

        # 2. BiSeNetV2 Track Segmentation Pass (512x1024)
        img_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (1024, 512), interpolation=cv2.INTER_LINEAR)
        norm = (img_resized / 255.0 - self.mean) / self.std
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(self.torch_device)

        with torch.no_grad():
            logits = self.seg_model(tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_full = cv2.resize(pred_mask.astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # Extract Vector Polygons for Track Bed (Class 1) and Rail Lines (Class 2)
        track_bed_mask = (pred_full == 1)
        rail_lines_mask = (pred_full == 2)

        track_bed_polys = self._extract_polygons_from_mask(track_bed_mask, min_area=100)
        rail_lines_polys = self._extract_polygons_from_mask(rail_lines_mask, min_area=40)

        # 3. Dual-Layer YOLO11m Obstacle Detection Pass
        obstacles = []
        custom_boxes_list = []

        # Layer 1: Custom Railway Sabotage & Obstacle Detector
        custom_res = self.custom_det.predict(
            source=enhanced_frame,
            imgsz=self.imgsz,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False
        )[0]

        if custom_res.boxes is not None:
            boxes = custom_res.boxes.cpu().numpy()
            for idx in range(len(boxes)):
                cls_id = int(boxes.cls[idx])
                conf = float(boxes.conf[idx])
                x1, y1, x2, y2 = boxes.xyxy[idx]
                cname = self.custom_names.get(cls_id, f"Obstacle_{cls_id}")

                box_tuple = (int(x1), int(y1), int(x2), int(y2))
                custom_boxes_list.append(box_tuple)
                obstacles.append({
                    "box": box_tuple,
                    "class_id": cls_id,
                    "class_name": cname,
                    "confidence": conf,
                    "source": "custom_railway"
                })

        # Layer 2: Complementary Base Foundation Detector (Wildlife, Baggage, General Transport)
        if self.base_det is not None:
            base_res = self.base_det.predict(
                source=enhanced_frame,
                imgsz=self.imgsz,
                conf=max(self.conf_thresh, 0.30),
                classes=list(BASE_COMPLEMENTARY_CLASSES.keys()),
                device=self.device,
                verbose=False
            )[0]

            if base_res.boxes is not None:
                b_boxes = base_res.boxes.cpu().numpy()
                for idx in range(len(b_boxes)):
                    b_cls_id = int(b_boxes.cls[idx])
                    b_conf = float(b_boxes.conf[idx])
                    bx1, by1, bx2, by2 = b_boxes.xyxy[idx]
                    b_box = (int(bx1), int(by1), int(bx2), int(by2))
                    b_cname = BASE_COMPLEMENTARY_CLASSES.get(b_cls_id, f"Object_{b_cls_id}")

                    # Check IoU overlap with custom detections to avoid duplicate boxes
                    is_duplicate = False
                    for cb in custom_boxes_list:
                        ix1 = max(b_box[0], cb[0])
                        iy1 = max(b_box[1], cb[1])
                        ix2 = min(b_box[2], cb[2])
                        iy2 = min(b_box[3], cb[3])
                        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        b_area = (b_box[2] - b_box[0]) * (b_box[3] - b_box[1])
                        c_area = (cb[2] - cb[0]) * (cb[3] - cb[1])
                        union_area = b_area + c_area - inter_area
                        if union_area > 0 and (inter_area / union_area) > 0.45:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        obstacles.append({
                            "box": b_box,
                            "class_id": 100 + b_cls_id,
                            "class_name": b_cname,
                            "confidence": b_conf,
                            "source": "base_foundation"
                        })

        # 4. Geometric Spatial Clearance Reasoning
        hazards, overall_status = self.hazard_analyzer.analyze(
            track_bed_polys=track_bed_polys,
            rail_lines_polys=rail_lines_polys,
            obstacles=obstacles,
            image_shape=(h_orig, w_orig)
        )

        # 5. Compute Frame Rate
        t_end = time.time()
        frame_time = t_end - t_start
        self.fps = 1.0 / max(frame_time, 0.001)

        # 6. Render Telemetry HUD
        rendered = self.visualizer.render(
            frame_bgr=enhanced_frame,
            track_bed_polys=track_bed_polys,
            rail_lines_polys=rail_lines_polys,
            hazards=hazards,
            overall_status=overall_status,
            fps=self.fps,
            sensor_mode=sensor_type,
            weather_status=weather_status,
            show_hud=show_hud
        )

        telemetry = {
            "fps": self.fps,
            "overall_status": overall_status,
            "weather_status": weather_status,
            "fog_score": fog_score,
            "num_hazards": len(hazards),
            "track_bed_found": len(track_bed_polys) > 0,
            "rail_lines_found": len(rail_lines_polys) > 0
        }

        return rendered, overall_status, hazards, telemetry
