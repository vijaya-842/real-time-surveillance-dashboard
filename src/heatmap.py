"""
Temporal movement heatmap and trajectory analytics.
Accumulates positional data over time with configurable decay.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


class HeatmapEngine:
    """
    Maintains a floating-point accumulation map of person positions.
    Each frame, existing heat decays and new detections add energy.
    """

    def __init__(self,
                 frame_h:     int   = 480,
                 frame_w:     int   = 640,
                 decay_rate:  float = 0.98,
                 blur_kernel: int   = 25,
                 colormap:    int   = cv2.COLORMAP_HOT,
                 alpha:       float = 0.55) -> None:

        self.h           = frame_h
        self.w           = frame_w
        self.decay_rate  = decay_rate
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.colormap    = colormap
        self.alpha       = alpha
        self._map        = np.zeros((frame_h, frame_w), dtype=np.float32)
        self._frame_cnt  = 0

    def update(self, tracks: list) -> None:
        """Decay existing heat and add new energy at track centroid positions."""
        self._map       *= self.decay_rate
        self._frame_cnt += 1

        for track in tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            cx = int(np.clip((x1 + x2) / 2, 0, self.w - 1))
            cy = int(np.clip((y1 + y2) / 2, 0, self.h - 1))

            # Gaussian splat — spread energy across a local region
            bw = max(int((x2 - x1) * 0.3), 10)
            bh = max(int((y2 - y1) * 0.3), 10)
            x1b = max(cx - bw, 0); x2b = min(cx + bw, self.w)
            y1b = max(cy - bh, 0); y2b = min(cy + bh, self.h)
            self._map[y1b:y2b, x1b:x2b] += 1.0

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Overlay the normalised, blurred heatmap on the given BGR frame."""
        if self._map.max() == 0:
            return frame

        normalised = (self._map / self._map.max() * 255).astype(np.uint8)
        blurred    = cv2.GaussianBlur(normalised,
                                      (self.blur_kernel, self.blur_kernel), 0)
        coloured   = cv2.applyColorMap(blurred, self.colormap)

        # Mask: only show heat where values are meaningful
        mask = (blurred > 10).astype(np.uint8)
        overlay = cv2.addWeighted(frame, 1 - self.alpha,
                                  coloured, self.alpha, 0)
        result = np.where(mask[..., None] == 1, overlay, frame)
        return result.astype(np.uint8)

    def reset(self) -> None:
        self._map[:] = 0
        self._frame_cnt = 0

    @property
    def stats(self) -> dict:
        return {
            "frames_processed": self._frame_cnt,
            "max_heat":         float(self._map.max()),
            "coverage_pct":     float((self._map > 0.1).mean() * 100),
        }


class TrajectoryRenderer:
    """Draws per-track movement trails on the video frame."""

    def __init__(self, max_tail: int = 40,
                 thickness: int = 2) -> None:
        self.max_tail  = max_tail
        self.thickness = thickness
        self._palette  = self._generate_palette(200)

    @staticmethod
    def _generate_palette(n: int) -> List[Tuple[int,int,int]]:
        np.random.seed(0)
        return [tuple(int(c) for c in np.random.randint(100, 255, 3))
                for _ in range(n)]

    def draw(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        out = frame.copy()
        for track in tracks:
            color = self._palette[track.track_id % len(self._palette)]
            pts   = [
                (int((b[0]+b[2])/2), int((b[1]+b[3])/2))
                for b in track.history[-self.max_tail:]
            ]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                t     = int(self.thickness * alpha)
                cv2.line(out, pts[i-1], pts[i], color, max(t, 1))
        return out
