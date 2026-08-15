"""
Drishti Kavach: Classic High-Contrast Track Visualizer & Clean Old-School Railway HUD

Design Philosophy:
  - Clean, minimal, old-school tactical locomotive HUD top header only.
  - High-Contrast Track Overlay:
      • Track Bed   : Vibrant Translucent Cyan-Blue (BGR: 255, 180, 0) [50% Alpha Blend]
      • Rail Lines  : Glowing Emerald Green (BGR: 0, 255, 100) with closed boundary
      • Obstacles   : Dynamic Three-Tier Badges (Red / Amber / Green) with ground anchor crosshairs
  - Zero clutter: No floating widgets or side cards on the main viewport.
"""

from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
from src.spatial_reasoning.hazard_analyzer import HazardAssessment


class DrishtiVisualizer:
    """
    Renders high-contrast railway track overlays and a clean, old-school top HUD header.
    """

    def __init__(self):
        # High-Contrast Track Colors (matching pristine_test)
        self.COLOR_TRACK_BED = (255, 180, 0)     # Translucent Cyan-Blue (BGR)
        self.COLOR_RAIL_LINES = (0, 255, 100)    # Glowing Emerald Green (BGR)
        self.COLOR_RAIL_BORDER = (0, 210, 80)    # Rail Outline (BGR)

        # Three-Tier Threat Colors
        self.COLOR_CRITICAL = (0, 0, 230)       # Vivid Red (BGR)
        self.COLOR_WARNING = (0, 210, 255)      # Amber / Gold (BGR)
        self.COLOR_SAFE = (40, 210, 60)         # Green (BGR)

        # Old-School HUD Colors
        self.HUD_BG = (12, 14, 18)              # Deep Charcoal Black (BGR)
        self.HUD_LINE = (0, 200, 230)           # Cyan Accent Line (BGR)
        self.TEXT_MAIN = (240, 245, 250)
        self.TEXT_MUTED = (160, 175, 190)

    def draw_hud_header(
        self,
        canvas: np.ndarray,
        overall_status: str,
        fps: float,
        sensor_mode: str,
        weather_status: str,
        hazards: List[HazardAssessment]
    ) -> np.ndarray:
        """
        Draws an old-school, large and commanding top status header with clean typography.
        """
        h, w = canvas.shape[:2]
        header_h = 80

        # 1. Top Header Background Strip
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), self.HUD_BG, -1)
        cv2.line(overlay, (0, header_h - 1), (w, header_h - 1), (50, 65, 80), 1)
        cv2.line(overlay, (0, header_h), (w, header_h), (75, 95, 120), 2)
        cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)

        # 2. Left: System Branding + Status Indicator Lamp
        lamp_col = self.COLOR_CRITICAL if overall_status == "CRITICAL" else (
            self.COLOR_WARNING if overall_status == "WARNING" else self.COLOR_SAFE
        )
        cv2.circle(canvas, (24, 40), 6, lamp_col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (24, 40), 8, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, "DRISHTI KAVACH", (44, 47),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 235, 255), 1, cv2.LINE_AA)

        # 3. Center: Old-School Tactical Safety Alert Box (Larger 46px Height)
        badge_w = 580
        badge_h = 46
        badge_x1 = (w - badge_w) // 2
        badge_y1 = 17
        badge_x2 = badge_x1 + badge_w
        badge_y2 = badge_y1 + badge_h

        if overall_status == "CRITICAL":
            box_bg = (0, 0, 185)
            box_border = (0, 0, 255)
            badge_text = "[ ! EMERGENCY BRAKE : OBSTACLE IN TRACK ! ]"
            text_color = (255, 255, 255)
        elif overall_status == "WARNING":
            box_bg = (0, 140, 205)
            box_border = (0, 220, 255)
            badge_text = "[ CAUTION : CLEARANCE ENVELOPE BREACH ]"
            text_color = (255, 255, 255)
        else:
            box_bg = (15, 70, 22)
            box_border = (40, 190, 55)
            badge_text = "[ TRACK STATUS : ALL CLEAR / NOMINAL ]"
            text_color = (100, 255, 120)

        cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), box_bg, -1)
        cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), box_border, 1)

        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
        tx = badge_x1 + (badge_w - tw) // 2
        ty = badge_y1 + (badge_h + th) // 2 - 1
        cv2.putText(canvas, badge_text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.65, text_color, 1, cv2.LINE_AA)

        # 4. Right: Telemetry (FPS, SENSOR, WEATHER)
        fps_str = f"FPS: {fps:4.1f}"
        sensor_str = f"SENSOR: {sensor_mode}"
        optics_str = f"OPTICS: {weather_status}"
        telemetry_str = f"{fps_str}  |  {sensor_str}  |  {optics_str}"

        (rw, rh), _ = cv2.getTextSize(telemetry_str, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 1)
        rx = w - rw - 24
        cv2.putText(canvas, telemetry_str, (rx, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.54, self.TEXT_MAIN, 1, cv2.LINE_AA)

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
        2. Clean obstacle bounding boxes with subtle badge
        3. Old-school sleek top HUD header
        """
        h, w = frame_bgr.shape[:2]
        canvas = frame_bgr.copy()
        track_overlay = frame_bgr.copy()

        # 1. Render Track Bed (Cyan-Blue 50% blend)
        for poly in track_bed_polys:
            if len(poly) >= 3:
                pts_np = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(track_overlay, [pts_np], self.COLOR_TRACK_BED)

        # 2. Render Rail Lines (Glowing Green)
        for poly in rail_lines_polys:
            if len(poly) >= 3:
                pts_np = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(track_overlay, [pts_np], self.COLOR_RAIL_LINES)
                cv2.polylines(track_overlay, [pts_np], True, self.COLOR_RAIL_BORDER, 2, cv2.LINE_AA)

        # 3. Alpha Blend Track Overlays
        cv2.addWeighted(track_overlay, 0.50, canvas, 0.50, 0, canvas)

        # 4. Render Detected Obstacles (Clean class name + confidence only)
        for hzd in hazards:
            x1, y1, x2, y2 = hzd.box_coords
            color = hzd.color_bgr

            # Solid Bounding Box with Clean 2px Line
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Ground Contact Anchor Reticle
            ax = int((x1 + x2) / 2)
            ay = int(y2)
            cv2.circle(canvas, (ax, ay), 4, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (ax, ay), 7, (255, 255, 255), 1, cv2.LINE_AA)

            # Clean Label: Just Class Name and Confidence Percentage
            label = f"{hzd.class_name} {hzd.confidence:.0%}"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.46, 1)
            by1 = max(0, y1 - lh - 8)
            by2 = y1
            bx1 = x1
            bx2 = x1 + lw + 12

            # Clean solid badge background in threat color
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
            cv2.putText(canvas, label, (bx1 + 6, by2 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Render Top HUD Header Only
        if show_hud:
            canvas = self.draw_hud_header(
                canvas,
                overall_status=overall_status,
                fps=fps,
                sensor_mode=sensor_mode,
                weather_status=weather_status,
                hazards=hazards
            )

        return canvas
