"""
Drishti Kavach: Spatial Hazard & Track Clearance Reasoning Engine

CLI Flags & Usage:
  Integrated into the Drishti Kavach pipeline. Computes polygon-level geometric
  intersections between segmented railway tracks and detected physical obstacles.

Three-Tier Threat Classification:
  - CRITICAL (In-Track)     : Obstacle footprint directly intersects Track Bed or Rail Lines
  - WARNING (Near-Track)    : Obstacle is within the dynamic lateral clearance buffer zone
  - SAFE (Off-Track)        : Obstacle is outside the railway clearance envelope
"""

import math
from typing import List, Dict, Tuple, Optional
import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union


class HazardAssessment:
    """Encapsulates the clearance reasoning result for a detected obstacle."""
    def __init__(
        self,
        box_coords: Tuple[int, int, int, int],
        class_id: int,
        class_name: str,
        confidence: float,
        threat_level: str,  # 'CRITICAL', 'WARNING', 'SAFE'
        distance_to_track_px: float,
        overlap_area_px: float,
        description: str,
        color_bgr: Tuple[int, int, int]
    ):
        self.box_coords = box_coords
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.threat_level = threat_level
        self.distance_to_track_px = distance_to_track_px
        self.overlap_area_px = overlap_area_px
        self.description = description
        self.color_bgr = color_bgr


class SpatialHazardAnalyzer:
    """
    Evaluates spatial clearance between segmented railway tracks and detected obstacles.
    Uses vector polygon geometry (Shapely) for exact footprint intersections.
    """

    def __init__(self, warning_buffer_pixels: float = 65.0):
        """
        Args:
            warning_buffer_pixels: Lateral safety envelope margin around track bed.
        """
        self.warning_buffer_pixels = warning_buffer_pixels

        # Color codes for threat levels
        self.COLOR_CRITICAL = (0, 0, 240)    # Bright Red (BGR)
        self.COLOR_WARNING = (0, 215, 255)   # Amber / Vivid Yellow (BGR)
        self.COLOR_SAFE = (50, 220, 50)      # Clean Green (BGR)

    def analyze(
        self,
        track_bed_polys: List[np.ndarray],
        rail_lines_polys: List[np.ndarray],
        obstacles: List[Dict],
        image_shape: Tuple[int, int]
    ) -> Tuple[List[HazardAssessment], str]:
        """
        Performs full geometric clearance reasoning.

        Args:
            track_bed_polys: List of Nx2 integer numpy coordinate arrays for track bed.
            rail_lines_polys: List of Nx2 integer numpy coordinate arrays for rail lines.
            obstacles: List of obstacle dicts with keys: 'box' (x1, y1, x2, y2), 'class_id', 'class_name', 'confidence'.
            image_shape: (height, width) of the current frame.

        Returns:
            Tuple of (List[HazardAssessment], overall_system_status: 'CRITICAL' | 'WARNING' | 'CLEAR')
        """
        h, w = image_shape[:2]

        # 1. Build Shapely Polygons for Track Bed & Rails
        shapely_track_beds = []
        for poly_pts in track_bed_polys:
            if len(poly_pts) >= 3:
                try:
                    p = Polygon(poly_pts)
                    if p.is_valid and p.area > 50:
                        shapely_track_beds.append(p)
                    elif not p.is_valid:
                        p_fixed = p.buffer(0)
                        if p_fixed.is_valid and p_fixed.area > 50:
                            shapely_track_beds.append(p_fixed)
                except Exception:
                    pass

        shapely_rail_lines = []
        for poly_pts in rail_lines_polys:
            if len(poly_pts) >= 3:
                try:
                    p = Polygon(poly_pts)
                    if p.is_valid and p.area > 20:
                        shapely_rail_lines.append(p)
                    elif not p.is_valid:
                        p_fixed = p.buffer(0)
                        if p_fixed.is_valid and p_fixed.area > 20:
                            shapely_rail_lines.append(p_fixed)
                except Exception:
                    pass

        # Unified Track Geometry
        unified_track = None
        all_track_elements = shapely_track_beds + shapely_rail_lines
        if all_track_elements:
            unified_track = unary_union(all_track_elements)

        # Build Warning Clearance Envelope (Track + lateral buffer)
        warning_zone = None
        if unified_track is not None and not unified_track.is_empty:
            warning_zone = unified_track.buffer(self.warning_buffer_pixels)

        # 2. Evaluate each detected obstacle against the clearance envelope
        assessments: List[HazardAssessment] = []
        has_critical = False
        has_warning = False

        for obs in obstacles:
            x1, y1, x2, y2 = obs["box"]
            cls_id = obs["class_id"]
            cname = obs.get("class_name", f"Obstacle_{cls_id}")
            conf = obs.get("confidence", 1.0)

            # Footprint: Bottom 25% of the bounding box represents ground contact point
            bw = x2 - x1
            bh = y2 - y1
            foot_y1 = y2 - (bh * 0.25)
            footprint_poly = box(x1, foot_y1, x2, y2)
            center_anchor = Point((x1 + x2) / 2.0, y2)

            threat_level = "SAFE"
            distance_to_track = 9999.0
            overlap_area = 0.0
            color = self.COLOR_SAFE
            description = f"{cname} (Safe Off-Track)"

            if unified_track is not None and not unified_track.is_empty:
                # Check for direct intersection with track bed or rails
                if unified_track.intersects(footprint_poly) or unified_track.contains(center_anchor):
                    threat_level = "CRITICAL"
                    overlap = unified_track.intersection(footprint_poly)
                    overlap_area = overlap.area
                    distance_to_track = 0.0
                    color = self.COLOR_CRITICAL
                    description = f"EMERGENCY: {cname} IN-TRACK ({conf:.0%})"
                    has_critical = True

                elif warning_zone is not None and (warning_zone.intersects(footprint_poly) or warning_zone.contains(center_anchor)):
                    threat_level = "WARNING"
                    distance_to_track = unified_track.distance(center_anchor)
                    color = self.COLOR_WARNING
                    description = f"CAUTION: {cname} NEAR-TRACK ({distance_to_track:.0f}px away)"
                    has_warning = True

                else:
                    distance_to_track = unified_track.distance(center_anchor)
                    threat_level = "SAFE"
                    color = self.COLOR_SAFE
                    description = f"{cname} (Safe Clearance)"

            assessments.append(
                HazardAssessment(
                    box_coords=(int(x1), int(y1), int(x2), int(y2)),
                    class_id=cls_id,
                    class_name=cname,
                    confidence=conf,
                    threat_level=threat_level,
                    distance_to_track_px=distance_to_track,
                    overlap_area_px=overlap_area,
                    description=description,
                    color_bgr=color
                )
            )

        # Determine overall system safety state
        if has_critical:
            overall_status = "CRITICAL"
        elif has_warning:
            overall_status = "WARNING"
        else:
            overall_status = "CLEAR"

        return assessments, overall_status
