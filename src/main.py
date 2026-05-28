"""
SmartSurveillance — Entry point.
Runs the full real-time detection + tracking + dashboard pipeline
on a video file or webcam stream.

Usage:
    python src/main.py --source 0              # webcam
    python src/main.py --source video.mp4      # video file
    python src/main.py --source video.mp4 --no-heatmap
    python src/main.py --source video.mp4 --save output.avi
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(source: str,
        config_path: str = "config/config.yaml",
        save_path:   str | None = None,
        no_heatmap:  bool = False,
        no_display:  bool = False) -> None:

    cfg = load_config(config_path)
    det_cfg  = cfg["detection"]
    trk_cfg  = cfg["tracking"]
    si_cfg   = cfg["siamese"]
    hm_cfg   = cfg["heatmap"]
    dash_cfg = cfg["dashboard"]

    # ── Lazy imports (heavy) ─────────────────────────────────────────────────
    from src.detector  import PersonDetector
    from src.tracker   import MultiObjectTracker
    from src.dashboard import SurveillanceDashboard

    detector = PersonDetector(
        confidence_threshold = det_cfg["confidence_threshold"],
        nms_iou_threshold    = det_cfg["nms_iou_threshold"],
        device               = det_cfg["device"],
    )
    tracker = MultiObjectTracker(
        max_age       = trk_cfg["max_age"],
        min_hits      = trk_cfg["min_hits"],
        iou_threshold = trk_cfg["iou_threshold"],
    )

    # ── Video source ─────────────────────────────────────────────────────────
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        log.error("Cannot open source: %s", source)
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    log.info("Source: %s | Resolution: %dx%d | FPS: %.1f",
             source, frame_w, frame_h, fps_src)

    dashboard = SurveillanceDashboard(
        frame_h           = frame_h,
        frame_w           = frame_w,
        show_heatmap      = not no_heatmap,
        show_trajectories = dash_cfg.get("show_trajectories", True),
        alert_cooldown    = dash_cfg.get("alert_cooldown_sec", 10),
    )

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(save_path, fourcc, fps_src, (frame_w, frame_h))
        log.info("Saving output to: %s", save_path)

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_num = 0
    t_start   = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect
            detections = detector.detect(frame)
            det_boxes  = [d.bbox for d in detections]

            # Track
            tracks = tracker.update(det_boxes)

            # Render dashboard
            vis = dashboard.render(frame, detections, tracks)

            if not no_display:
                cv2.imshow(dash_cfg["window_title"], vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if key == ord("r"):
                    dashboard.heatmap_engine.reset()
                    log.info("Heatmap reset")

            if writer:
                writer.write(vis)

            frame_num += 1

    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        log.info("Processed %d frames in %.1fs (%.1f FPS avg)",
                 frame_num, elapsed, frame_num / max(elapsed, 1))
        log.info("Unique persons detected: %d",
                 len(dashboard._known_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     default="0")
    parser.add_argument("--config",     default="config/config.yaml")
    parser.add_argument("--save",       default=None)
    parser.add_argument("--no-heatmap", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    run(args.source, args.config, args.save, args.no_heatmap, args.no_display)
