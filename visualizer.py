"""
Visualizer — composes the final annotated output frame.

Layers (bottom to top):
  1. Original frame
  2. Green traversable path overlay (semi-transparent)
  3. Path edge highlighting
  4. HUD (road coverage, model info)
"""

import cv2
import numpy as np
from typing import List, Optional


class Visualizer:

    def __init__(self, config: dict):
        self.cfg = config
        self.path_color_bgr   = config.get("path_color_bgr",  (0, 220, 80))    # green
        self.path_alpha       = config.get("path_alpha",       0.40)
        self.path_edge_color  = config.get("path_edge_bgr",   (0, 255, 120))
        self.show_seg_overlay = config.get("show_seg_classes", False)
        self.font             = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale       = config.get("font_scale", 0.55)

    # ------------------------------------------------------------------
    # Main draw function
    # ------------------------------------------------------------------

    def draw(self,
             frame:       np.ndarray,
             road_mask:   np.ndarray,
             seg_map:     Optional[np.ndarray] = None,
             class_names: Optional[List[str]]  = None) -> np.ndarray:

        H, W = frame.shape[:2]
        output = frame.copy()

        # Layer 1: Traversable path overlay
        output = self._draw_path_overlay(output, road_mask)

        # Layer 2: HUD
        output = self._draw_hud(output, road_mask)

        return output

    # ------------------------------------------------------------------
    # Path overlay
    # ------------------------------------------------------------------

    def _draw_path_overlay(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        overlay[mask > 0] = self.path_color_bgr

        # Find contours for edge highlighting
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, self.path_edge_color, 2)

        return cv2.addWeighted(overlay, self.path_alpha, frame, 1 - self.path_alpha, 0)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(self, frame: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
        H, W = frame.shape[:2]

        # Road coverage percentage
        road_pct = road_mask.sum() / (255 * H * W) * 100

        lines = [
            f"Road: {road_pct:.1f}%",
        ]

        y = H - (len(lines) * 22 + 12)
        bg = np.zeros((len(lines) * 22 + 12, 220, 3), dtype=np.uint8)
        bg[:] = (20, 20, 20)

        # Ensure we don't go out of bounds
        if y - 6 >= 0 and y + len(lines) * 22 + 6 <= H:
            frame[y - 6:y + len(lines) * 22 + 6, 6:226] = cv2.addWeighted(
                bg, 0.65,
                frame[y - 6:y + len(lines) * 22 + 6, 6:226],
                0.35, 0
            )

        for i, line in enumerate(lines):
            y_pos = y + i * 22 + 16
            if y_pos + 10 <= H:
                cv2.putText(frame, line, (12, y_pos),
                            self.font, 0.48, (220, 255, 200), 1, cv2.LINE_AA)

        # Top-left: model indicator
        cv2.putText(frame, "SegFormer-B5 | Path Detection",
                    (8, 22), self.font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        return frame