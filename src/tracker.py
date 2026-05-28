"""
SORT-style multi-object tracker using Kalman filters for motion prediction
and IoU-based Hungarian algorithm assignment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)


@dataclass
class KalmanTrack:
    """Single object track with Kalman-filter state [x, y, w, h, vx, vy, vw, vh]."""

    track_id:   int
    bbox:       np.ndarray          # current [x1,y1,x2,y2]
    hits:       int = 1
    no_match:   int = 0
    confirmed:  bool = False
    history:    List[np.ndarray] = field(default_factory=list)

    # Kalman state: [cx, cy, w, h, vcx, vcy, vw, vh]
    _state:     Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        cx, cy, w, h = self._bbox_to_center(self.bbox)
        self._state  = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        self.history.append(self.bbox.copy())

    @staticmethod
    def _bbox_to_center(bbox: np.ndarray) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

    @staticmethod
    def _center_to_bbox(cx, cy, w, h) -> np.ndarray:
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    def predict(self) -> np.ndarray:
        """Constant-velocity prediction step."""
        self._state[:4] += self._state[4:]
        self.bbox = self._center_to_bbox(*self._state[:4])
        return self.bbox

    def update(self, bbox: np.ndarray) -> None:
        """Measurement update — simple weighted blend."""
        cx, cy, w, h = self._bbox_to_center(bbox)
        alpha = 0.6
        self._state[:4] = (alpha * np.array([cx, cy, w, h]) +
                           (1 - alpha) * self._state[:4])
        self._state[4:] = alpha * (np.array([cx, cy, w, h]) - self._state[:4])
        self.bbox   = self._center_to_bbox(*self._state[:4])
        self.hits  += 1
        self.no_match = 0
        self.history.append(bbox.copy())
        if len(self.history) > 60:
            self.history.pop(0)


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix = max(0, min(xa2, xb2) - max(xa1, xb1))
    iy = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter = ix * iy
    if inter == 0:
        return 0.0
    union = (xa2-xa1)*(ya2-ya1) + (xb2-xb1)*(yb2-yb1) - inter
    return inter / union


class MultiObjectTracker:
    """
    Tracks multiple people across frames using Kalman prediction +
    Hungarian algorithm assignment on IoU cost matrix.
    """

    def __init__(self,
                 max_age:       int   = 30,
                 min_hits:      int   = 3,
                 iou_threshold: float = 0.3) -> None:
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self._tracks:      Dict[int, KalmanTrack] = {}
        self._next_id:     int = 1

    @property
    def active_tracks(self) -> List[KalmanTrack]:
        return [t for t in self._tracks.values() if t.confirmed]

    def update(self, detections: List[np.ndarray]) -> List[KalmanTrack]:
        """
        Update tracker state with new detections.
        Returns confirmed tracks with updated positions.
        """
        # Predict all existing tracks
        predicted = {tid: t.predict() for tid, t in self._tracks.items()}

        if not detections:
            self._age_tracks()
            return self.active_tracks

        if not self._tracks:
            for det in detections:
                self._tracks[self._next_id] = KalmanTrack(self._next_id, det)
                self._next_id += 1
            return self.active_tracks

        # Build IoU cost matrix
        track_ids = list(self._tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1 - _iou(self._tracks[tid].bbox, det)

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_tracks = set()
        matched_dets   = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < (1 - self.iou_threshold):
                tid = track_ids[r]
                self._tracks[tid].update(detections[c])
                matched_tracks.add(tid)
                matched_dets.add(c)

        # Spawn new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._tracks[self._next_id] = KalmanTrack(self._next_id, det)
                self._next_id += 1

        # Age unmatched tracks
        for tid in list(self._tracks.keys()):
            if tid not in matched_tracks:
                self._tracks[tid].no_match += 1
                if self._tracks[tid].no_match > self.max_age:
                    del self._tracks[tid]
                    continue
            if self._tracks[tid].hits >= self.min_hits:
                self._tracks[tid].confirmed = True

        return self.active_tracks

    def _age_tracks(self) -> None:
        for tid in list(self._tracks.keys()):
            self._tracks[tid].no_match += 1
            if self._tracks[tid].no_match > self.max_age:
                del self._tracks[tid]
