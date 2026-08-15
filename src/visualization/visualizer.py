"""
Drishti Kavach: High-Contrast Track Visualizer & Real-Time Railway HUD Dashboard

Features:
  - Exact Pristine Color Rendering:
      • Track Bed   : Vibrant Translucent Cyan-Blue (BGR: 255, 180, 0) [50% Alpha]
      • Rail Lines  : Glowing Emerald Green (BGR: 0, 255, 100)
      • Obstacles   : Dynamic Three-Tier Badges (Red / Amber / Green)
  - Glassmorphic Top Status Telemetry Bar with ATP Kavach Emergency Brake Warnings
  - Real-Time FPS, Sensor Channel Indicator, and Weather Status Telemetry
"""

from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
from src.spatial_reasoning.hazard_analyzer import HazardAssessment


class DrishtiVisualizer:
    """
    Renders high-contrast track segmentation masks, obstacle bounding boxes,
    and the real-time Kavach ATP telemetry HUD.
    """

    def __init__(self):
        # Exact Colors matching pristine_test
        self.COLOR_TRACK_BED = (255, 180, 0)     # Pristine Cyan-Blue (BGR)
        self.COLOR_RAIL_LINES = (0, 255, 100)    # Glowing Green (BGR)
        self.COLOR_RAIL_BORDER = (0, 210, 80)    # Rail Outline (BGR)

        # Three-Tier Threat Colors
        self.COLOR_CRITICAL = (0, 0, 240)       # Vivid Red (BGR)
        self.COLOR_WARNING = (0, 215, 255)      # Amber / Gold (BGR)
        self.COLOR_SAFE = (50, 220, 50)         # Green (BGR)

        # HUD Palette
        self.HUD_BG = (18, 20, 24)              # Dark Navy Slate (BGR)
        self.HUD_BORDER = (45, 52, 64)          # Subtle Border (BGR)
        self.TEXT_WHITE = (245, 245, 250)
        self.TEXT_MUTED = (160, 170, 185)

    def draw_hud_banner(
        self,
        canvas: np.ndarray,
        overall_status: str,
        fps: float,
        sensor_mode: str,
        weather_status: str,
        hazards: List[HazardAssessment]
    ) -> np.ndarray:
        """Draws top glassmorphism status bar with dynamic safety state."""
        h, w = canvas.shape[:2]
        banner_h = 60

        # Semi-transparent top banner background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), self.HUD_BG, -1)
        cv2.line(overlay, (0, banner_h), (w, banner_h), self.HUD_BORDER, 2)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)

        # Left: Branding
        cv2.putText(canvas, "DRISHTI KAVACH", (20, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 235, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "ATP OPTICAL PERCEPTION", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.TEXT_MUTED, 1, cv2.LINE_AA)

        # Center: Safety Status Badge
        badge_w = 420
        badge_h = 42
        badge_x1 = (w - badge_w) // 2
        badge_y1 = 9
        badge_x2 = badge_x1 + badge_w
        badge_y2 = badge_y1 + badge_h

        if overall_status == "CRITICAL":
            badge_color = self.COLOR_CRITICAL
            badge_text = "EMERGENCY: OBSTACLE IN TRACK [BRAKE]"
            border_color = (120, 120, 255)
        elif overall_status == "WARNING":
            badge_color = self.COLOR_WARNING
            badge_text = "CAUTION: CLEARANCE BREACH"
            border_color = (100, 235, 255)
        else:
            badge_color = self.COLOR_SAFE
            badge_text = "TRACK CLEAR - ALL CLEAR"
            border_color = (100, 255, 120)

        cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_color, -1)
        cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), border_color, 2)
        
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.60, 2)
        text_x = badge_x1 + (badge_w - tw) // 2
        text_y = badge_y1 + (badge_h + th) // 2 - 1
        cv2.putText(canvas, badge_text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

        # Right: Telemetry (FPS, Sensor, Weather)
        right_x = w - 310
        cv2.putText(canvas, f"FPS: {fps:.1f}", (right_x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"SENSOR: {sensor_mode}", (right_x + 95, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"OPTICS: {weather_status}", (right_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_MUTED, 1, cv2.LINE_AA)

        return canvas

    def draw_telemetry_card(
        self,
        canvas: np.ndarray,
        track_detected: bool,
        hazards: List[HazardAssessment]
    ) -> np.ndarray:
        """Draws bottom-left telemetry diagnostics card."""
        h, w = canvas.shape[:2]
        card_w = 340
        card_h = 110 + min(len(hazards), 3) * 22
        x1 = 20
        y1 = h - card_h - 20
        x2 = x1 + card_w
        y2 = h - 20

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.HUD_BG, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.HUD_BORDER, 2)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        cv2.putText(canvas, "CLEARANCE ENVELOPE TELEMETRY", (x1 + 14, y1 + 24), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 235, 255), 1, cv2.LINE_AA)
        
        track_str = "TRACK GEOMETRY: LOCKED" if track_detected else "TRACK GEOMETRY: ACQUIRING"
        track_col = (50, 220, 50) if track_detected else (0, 180, 255)
        cv2.putText(canvas, track_str, (x1 + 14, y1 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, track_col, 1, cv2.LINE_AA)

        crit_count = sum(1 for hzd in hazards if hzd.threat_level == "CRITICAL")
        warn_count = sum(1 for hzd in hazards if hzd.threat_level == "WARNING")
        
        cv2.putText(canvas, f"CRITICAL HAZARDS: {crit_count}", (x1 + 14, y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_CRITICAL if crit_count > 0 else self.TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"WARNING HAZARDS:  {warn_count}", (x1 + 180, y1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_WARNING if warn_count > 0 else self.TEXT_MUTED, 1, cv2.LINE_AA)

        # List top detected items
        cur_y = y1 + 96
        for idx, hzd in enumerate(hazards[:3]):
            desc = f"• {hzd.class_name} [{hzd.threat_level}] - {hzd.confidence:.0%}"
            cv2.putText(canvas, desc, (x1 + 14, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, hzd.color_bgr, 1, cv2.LINE_AA)
            cur_y += 20

        return canvas

    def render(
        self,
        frame_bgr: np.ndarray,
        track_bed_polys: List[np.ndarray],
        rail_lines_polys: List[np.ndarray],
        hazards: List[HazardAssessment],
        overall_status: str,
        fps: float = 0.0,
        sensor_mode: str = "DAYLIGHT RGB",
        weather_status: str = "CLEAR",
        show_hud: bool = True
    ) -> np.ndarray:
        """
        Renders full visualizer output:
        1. High-contrast track bed & rail lines masks
        2. Hazard bounding boxes with dynamic threat badges
        3. Real-time Kavach telemetry HUD
        """
        h, w = frame_bgr.shape[:2]
        canvas = frame_bgr.copy()
        track_overlay = frame_bgr.copy()
        has_track = False

        # 1. Render Track Bed (Cyan-Blue)
        for poly in track_bed_polys:
            if len(poly) >= 3:
                pts_np = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(track_overlay, [pts_np], self.COLOR_TRACK_BED)
                has_track = True

        # 2. Render Rail Lines (Glowing Green)
        for poly in rail_lines_polys:
            if len(poly) >= 3:
                pts_np = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(track_overlay, [pts_np], self.COLOR_RAIL_LINES)
                cv2.polylines(track_overlay, [pts_np], True, self.COLOR_RAIL_BORDER, 2, cv2.LINE_AA)
                has_track = True

        # Single 50/50 blend for crisp vibrancy
        if has_track:
            cv2.addWeighted(track_overlay, 0.50, canvas, 0.50, 0, canvas)

        # 3. Render Obstacle Bounding Boxes & Dynamic Badges
        for hzd in hazards:
            x1, y1, x2, y2 = hzd.box_coords
            color = hzd.color_bgr
            
            # Thick bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

            # Ground contact anchor line
            anchor_x = (x1 + x2) // 2
            cv2.circle(canvas, (anchor_x, y2), 5, color, -1)
            cv2.circle(canvas, (anchor_x, y2), 7, (255, 255, 255), 1)

            # Badge Text
            if hzd.threat_level == "CRITICAL":
                badge_text = f"CRITICAL: {hzd.class_name} [{hzd.confidence:.0%}]"
            elif hzd.threat_level == "WARNING":
                badge_text = f"WARNING: {hzd.class_name} [{hzd.distance_to_track_px:.0f}px]"
            else:
                badge_text = f"{hzd.class_name} [{hzd.confidence:.0%}]"

            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.58, 1)
            
            badge_y1 = max(0, y_min if 'y_min' in locals() else y1 - th - 12, y1 - th - 12)
            badge_y2 = max(th + 12, y1)
            badge_x2 = min(w, x1 + tw + 16)

            # Solid color badge background + white border
            cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), (255, 255, 255), 1)

            # White text
            cv2.putText(canvas, badge_text, (x1 + 8, badge_y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Render Telemetry HUD
        if show_hud:
            self.draw_hud_banner(canvas, overall_status, fps, sensor_mode, weather_status, hazards)
            self.draw_telemetry_card(canvas, has_track, hazards)

        return canvas
