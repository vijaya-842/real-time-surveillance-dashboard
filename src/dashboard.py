"""
Real-time surveillance intelligence dashboard.
Renders detection boxes, track IDs, trajectories, heatmap overlay,
FPS counter, and alert notifications on the live video frame.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Set

import cv2
import numpy as np

from src.detector import Detection
from src.heatmap  import HeatmapEngine, TrajectoryRenderer


class SurveillanceDashboard:
    """Composites all visual layers onto the live video frame."""

    FONT       = cv2.FONT_HERSHEY_DUPLEX
    FONT_SCALE = 0.55
    THICKNESS  = 1

    def __init__(self,
                 frame_h:          int   = 480,
                 frame_w:          int   = 640,
                 show_heatmap:     bool  = True,
                 show_trajectories:bool  = True,
                 alert_cooldown:   float = 10.0) -> None:

        self.heatmap_engine = HeatmapEngine(frame_h, frame_w)
        self.trajectory_r   = TrajectoryRenderer()
        self.show_heatmap   = show_heatmap
        self.show_traj      = show_trajectories
        self.alert_cooldown = alert_cooldown

        self._fps_buf: deque  = deque(maxlen=30)
        self._last_t:  float  = time.time()
        self._alerts:  Dict[int, float] = {}   # track_id -> last alert time
        self._known_ids: Set[int] = set()
        self._alert_log: list = []

    # ── Per-frame render ──────────────────────────────────────────────────────

    def render(self, frame: np.ndarray, detections: List[Detection],
               tracks: list) -> np.ndarray:

        self._update_fps()
        self.heatmap_engine.update(tracks)

        out = frame.copy()

        if self.show_heatmap:
            out = self.heatmap_engine.render(out)

        if self.show_traj:
            out = self.trajectory_r.draw(out, tracks)

        # Draw raw detection boxes (light)
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            cv2.rectangle(out, (x1,y1), (x2,y2), (200,200,200), 1)

        # Draw confirmed track boxes + IDs
        for track in tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            color = self._track_color(track.track_id)
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
            label = f"ID {track.track_id}"
            (tw, th), _ = cv2.getTextSize(label, self.FONT,
                                          self.FONT_SCALE, self.THICKNESS)
            cv2.rectangle(out, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
            cv2.putText(out, label, (x1+2, y1-4),
                        self.FONT, self.FONT_SCALE, (255,255,255), self.THICKNESS)
            self._check_alert(track)

        self._draw_hud(out, len(detections), len(tracks))
        self._draw_alerts(out)
        return out

    # ── HUD ──────────────────────────────────────────────────────────────────

    def _draw_hud(self, frame: np.ndarray, n_det: int, n_trk: int) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (220, 80), (20,20,20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        fps   = self._current_fps()
        lines = [
            f"FPS  : {fps:.1f}",
            f"Dets : {n_det}",
            f"Trks : {n_trk}",
            f"Cvrg : {self.heatmap_engine.stats['coverage_pct']:.1f}%",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (8, 18 + i*16),
                        self.FONT, 0.48, (0,255,180), 1)

    def _draw_alerts(self, frame: np.ndarray) -> None:
        h = frame.shape[0]
        for i, msg in enumerate(self._alert_log[-3:]):
            y = h - 15 - i * 20
            cv2.putText(frame, msg, (8, y),
                        self.FONT, 0.45, (0,60,255), 1)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_alert(self, track) -> None:
        now = time.time()
        if track.track_id not in self._known_ids:
            self._known_ids.add(track.track_id)
            self._alerts[track.track_id] = now
            msg = f"[ALERT] New person detected — ID {track.track_id}"
            self._alert_log.append(msg)
        elif (now - self._alerts.get(track.track_id, 0)) > self.alert_cooldown:
            self._alerts[track.track_id] = now

    def _update_fps(self) -> None:
        now = time.time()
        self._fps_buf.append(1.0 / max(now - self._last_t, 1e-6))
        self._last_t = now

    def _current_fps(self) -> float:
        return float(np.mean(self._fps_buf)) if self._fps_buf else 0.0

    @staticmethod
    def _track_color(track_id: int) -> tuple:
        np.random.seed(track_id * 37)
        return tuple(int(c) for c in np.random.randint(80, 240, 3))
