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


class DrishtiEngine:
    """
    Complete real-time perception pipeline for Indian Railways Optical Kavach.
    """

    def __init__(
        self,
        seg_model_path: str = "models/RailDrishti_Seg_Universal.pth",
        det_model_path: str = "models/best_yolo11m_raildrishti.pt",
        conf_thresh: float = 0.35,
        imgsz: int = 1024,
        device: Optional[str] = None,
        weather_mode: str = "auto",
        warning_buffer_px: float = 65.0
    ):
        self.seg_model_path = seg_model_path
        self.det_model_path = det_model_path
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.weather_mode = weather_mode

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
        print(" 🛡️  INITIALIZING DRISHTI KAVACH REAL-TIME ENGINE (DECOUPLED)")
        print("=" * 75)
        print(f" • Segmentation Model: {self.seg_model_path}")
        print(f" • Obstacle Detector:  {self.det_model_path}")
        print(f" • Hardware Device:    {self.device}")
        print(f" • Target Resolution:  {imgsz}x{imgsz}")
        print(f" • Confidence Cutoff:  {conf_thresh:.2f}")
        print(f" • Weather Optimizer:  {weather_mode.upper()}")
        print("=" * 75)

        # 1. Load BiSeNetV2 Semantic Segmentation Model
        self.seg_model = BiSeNetV2(num_classes=3, is_training=False).to(self.torch_device)
        
        # Check primary universal weights, then base weights fallback
        actual_seg_path = self.seg_model_path
        if not os.path.exists(actual_seg_path):
            fallback_path = "models/best_bisenetv2_raildrishti.pth"
            if os.path.exists(fallback_path):
                actual_seg_path = fallback_path
                print(f"[*] Using base segmentation weights: {fallback_path}")

        if os.path.exists(actual_seg_path):
            state_dict = torch.load(actual_seg_path, map_location=self.torch_device)
            infer_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux")}
            self.seg_model.load_state_dict(infer_state_dict, strict=False)
            print(f"[+] Loaded BiSeNetV2 Track Model from: {actual_seg_path}")
        else:
            print(f"[!] Warning: No segmentation checkpoint found at {actual_seg_path}.")

        self.seg_model.eval()

        # 2. Load YOLO11m Obstacle Detector
        if os.path.exists(self.det_model_path):
            self.det_model = YOLO(self.det_model_path)
            print(f"[+] Loaded YOLO11m Obstacle Model from: {self.det_model_path}")
        else:
            print(f"[*] Obstacle weights '{self.det_model_path}' not found yet. Using 'yolo11m.pt' baseline.")
            self.det_model = YOLO("yolo11m.pt")

        self.names = OBSTACLE_CLASSES

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

        # 3. YOLO11m Obstacle Detection Pass
        det_results = self.det_model.predict(
            source=enhanced_frame,
            imgsz=self.imgsz,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False
        )[0]

        obstacles = []
        if det_results.boxes is not None:
            boxes = det_results.boxes.cpu().numpy()
            for idx in range(len(boxes)):
                cls_id = int(boxes.cls[idx])
                conf = float(boxes.conf[idx])
                x1, y1, x2, y2 = boxes.xyxy[idx]
                cname = self.names.get(cls_id, f"Obstacle_{cls_id}")

                obstacles.append({
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "class_id": cls_id,
                    "class_name": cname,
                    "confidence": conf
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
