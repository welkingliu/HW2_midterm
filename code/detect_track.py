"""
Task 2: Scene Object Detection, Multi-Object Tracking & Cross-Line Counting
Features:
- YOLOv8 fine-tuning on Road Vehicle Images Dataset
- model.track() for stable Tracking IDs
- Virtual cross-line counting with ID deduplication
- Occlusion frame export for ID-switch analysis
"""

import os
import cv2
import yaml
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional


def _json_default(obj):
    """Convert numpy values before writing JSON summaries."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# ─────────────────────────────────────────────────────────────────────────────
#  Dependencies:  pip install ultralytics opencv-python-headless
# ─────────────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Run: pip install ultralytics")


# ─── 1. Dataset Config Generator ─────────────────────────────────────────────

def create_dataset_yaml(dataset_root: str, output_path: str = "vehicle_dataset.yaml"):
    """
    Road Vehicle Images Dataset YAML for YOLOv8.
    Assumes YOLO-format directory:
        dataset_root/
            images/train/  images/val/
            labels/train/  labels/val/
    Adjust class names to match your dataset labels.
    """
    for name in ("data.yaml", "data_1.yaml", "dataset.yaml"):
        existing = Path(dataset_root) / name
        if existing.is_file():
            print(f"[Dataset YAML] using existing file: {existing}")
            return str(existing)

    config = {
        "path": os.path.abspath(dataset_root),
        "train": "images/train",
        "val":   "images/val",
        "nc": 4,
        "names": {
            0: "car",
            1: "truck",
            2: "bus",
            3: "motorcycle",
        },
    }
    with open(output_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"[Dataset YAML] saved to {output_path}")
    return output_path


# ─── 2. Fine-tuning ───────────────────────────────────────────────────────────

def finetune_yolo(
    dataset_yaml: str,
    model_size: str = "yolov8n",     # yolov8n / yolov8s / yolov8m
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    output_dir: str = "./outputs/task2",
):
    """Fine-tune YOLOv8 on the vehicle dataset."""
    model = YOLO(f"{model_size}.pt")   # downloads pretrained weights automatically

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=output_dir,
        name="vehicle_finetune",
        exist_ok=True,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        augment=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0,
        save_period=10,
        verbose=True,
    )

    save_dir = Path(getattr(results, "save_dir", Path(output_dir) / "vehicle_finetune"))
    best_model_path = save_dir / "weights" / "best.pt"
    if not best_model_path.exists():
        last_model_path = save_dir / "weights" / "last.pt"
        if last_model_path.exists():
            best_model_path = last_model_path
        else:
            raise FileNotFoundError(
                f"Could not find trained weights under {save_dir / 'weights'}"
            )
    best_model_path = str(best_model_path.resolve())
    print(f"[Fine-tune] Best model: {best_model_path}")
    return best_model_path


# ─── 3. Cross-Line Counter ────────────────────────────────────────────────────

class CrossLineCounter:
    """
    Counts unique objects that cross a virtual line.

    The line is defined by two points: (x1, y1) → (x2, y2).
    We use the sign of the cross-product between the line direction vector
    and the vector from line start to object center to determine which side
    an object is on. A sign change across frames = crossing event.
    """

    def __init__(self, pt1: tuple, pt2: tuple):
        self.pt1 = np.array(pt1, dtype=float)
        self.pt2 = np.array(pt2, dtype=float)
        self.prev_side: dict[int, int] = {}     # track_id → last side (+1 or -1)
        self.crossed_ids: set[int] = set()       # IDs that already crossed
        self.count: int = 0

    def _side(self, cx: float, cy: float) -> int:
        """Returns +1 or -1 depending on which side of the line (cx,cy) is on."""
        d = self.pt2 - self.pt1
        v = np.array([cx, cy]) - self.pt1
        cross = d[0] * v[1] - d[1] * v[0]
        return 1 if cross >= 0 else -1

    def update(self, track_id: int, cx: float, cy: float) -> bool:
        """
        Call once per detection per frame.
        Returns True if this update triggered a new crossing event.
        """
        current_side = self._side(cx, cy)
        crossed = False
        if track_id in self.prev_side:
            if (self.prev_side[track_id] != current_side
                    and track_id not in self.crossed_ids):
                self.count += 1
                self.crossed_ids.add(track_id)
                crossed = True
        self.prev_side[track_id] = current_side
        return crossed

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Overlay the counting line and counter on a frame."""
        x1, y1 = map(int, self.pt1)
        x2, y2 = map(int, self.pt2)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(frame, f"Count: {self.count}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        return frame


# ─── 4. Occlusion Frame Extractor ────────────────────────────────────────────

class OcclusionAnalyzer:
    """
    Detects potential ID-switch / occlusion events by monitoring
    IoU overlap between bounding boxes and sudden ID re-appearance gaps.
    Saves 3-4 consecutive frames when such events are detected.
    """

    def __init__(
        self,
        output_dir: str,
        iou_thresh: float = 0.3,
        gap_thresh: int = 5,
        event_cooldown: int = 30,
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.iou_thresh = iou_thresh
        self.gap_thresh = gap_thresh
        self.event_cooldown = event_cooldown
        self.last_seen: dict[int, int] = {}     # track_id → last frame index
        self.frame_buffer: list = []             # rolling buffer of last 4 frames
        self.saved_events = 0
        self.MAX_EVENTS = 4
        self.last_event_frame = -10**9

    @staticmethod
    def _iou(box1, box2) -> float:
        """box = [x1, y1, x2, y2]"""
        xi1 = max(box1[0], box2[0]); yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2]); yi2 = min(box1[3], box2[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, frame: np.ndarray, frame_idx: int,
               track_ids: list, boxes: list):
        """
        frame:      annotated BGR frame
        track_ids:  list of int IDs visible this frame
        boxes:      parallel list of [x1,y1,x2,y2] xyxy boxes
        """
        # Rolling 4-frame buffer
        self.frame_buffer.append((frame_idx, frame.copy()))
        if len(self.frame_buffer) > 4:
            self.frame_buffer.pop(0)

        occlusion_detected = False

        # 1. Check pairwise IoU for dense-overlap
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if self._iou(boxes[i], boxes[j]) > self.iou_thresh:
                    occlusion_detected = True
                    break
            if occlusion_detected:
                break

        # 2. Check for ID re-appearance after gap (possible ID switch)
        for tid in track_ids:
            if tid in self.last_seen:
                gap = frame_idx - self.last_seen[tid]
                if gap > self.gap_thresh:
                    occlusion_detected = True
            self.last_seen[tid] = frame_idx

        # Save buffer when event detected
        can_save = (frame_idx - self.last_event_frame) >= self.event_cooldown
        if occlusion_detected and can_save and self.saved_events < self.MAX_EVENTS:
            event_dir = os.path.join(self.output_dir, f"event_{self.saved_events:02d}")
            os.makedirs(event_dir, exist_ok=True)
            for buf_idx, (fidx, fimg) in enumerate(self.frame_buffer):
                cv2.imwrite(
                    os.path.join(event_dir, f"frame_{fidx:05d}.jpg"), fimg
                )
            print(f"[OcclusionAnalyzer] Saved event {self.saved_events} "
                  f"at frame {frame_idx}")
            self.saved_events += 1
            self.last_event_frame = frame_idx


# ─── 5. Full MOT + Counting Pipeline ─────────────────────────────────────────

def run_tracking_pipeline(
    model_path: str,
    video_path: str,
    output_dir: str = "./outputs/task2",
    line_pt1: tuple = (0, 0),       # will be auto-set to frame midpoint if (0,0)
    line_pt2: tuple = (0, 0),
    conf_thresh: float = 0.35,
    iou_thresh: float = 0.45,
    event_cooldown: int = 30,
):
    """
    Full pipeline:
      1. Load fine-tuned YOLOv8
      2. Run model.track() on each frame
      3. Cross-line counting
      4. Occlusion frame export
      5. Annotated video output
    """
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)
    cap   = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {width}x{height}  {fps:.1f}fps  {total_frames} frames")

    # Auto line: horizontal midline at 40% from top
    if line_pt1 == (0, 0) and line_pt2 == (0, 0):
        y_line = int(height * 0.40)
        line_pt1 = (0, y_line)
        line_pt2 = (width, y_line)
    print(f"[Counter] Line  {line_pt1} → {line_pt2}")

    counter  = CrossLineCounter(line_pt1, line_pt2)
    analyzer = OcclusionAnalyzer(
        output_dir=os.path.join(output_dir, "occlusion_frames"),
        event_cooldown=event_cooldown,
    )

    out_path = os.path.join(output_dir, "tracked_output.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Track history for trajectory visualisation
    track_history: dict[int, list] = defaultdict(list)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── YOLOv8 Track ─────────────────────────────────────────────────────
        results = model.track(
            frame,
            persist=True,
            conf=conf_thresh,
            iou=iou_thresh,
            tracker="bytetrack.yaml",   # ByteTrack for robust MOT
            verbose=False,
        )

        annotated = frame.copy()
        track_ids_this_frame = []
        boxes_this_frame = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            track_ids  = results[0].boxes.id.cpu().numpy().astype(int)
            classes    = results[0].boxes.cls.cpu().numpy().astype(int)
            confs      = results[0].boxes.conf.cpu().numpy()
            class_names = model.names

            for box, tid, cls, conf in zip(boxes_xyxy, track_ids, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Cross-line counting
                crossed = counter.update(tid, cx, cy)

                # Track history (tail visualisation)
                track_history[tid].append((cx, cy))
                if len(track_history[tid]) > 30:
                    track_history[tid].pop(0)

                # Colour: flash red on crossing
                color = (0, 0, 255) if crossed else (0, 200, 0)

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{tid} {class_names[cls]} {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # Draw trajectory tail
                pts = track_history[tid]
                for k in range(1, len(pts)):
                    cv2.line(annotated, pts[k-1], pts[k], (255, 165, 0), 1)

                track_ids_this_frame.append(tid)
                boxes_this_frame.append([x1, y1, x2, y2])

        # Overlay line and count
        annotated = counter.draw(annotated)

        # Frame index overlay
        cv2.putText(annotated, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Occlusion analysis
        analyzer.update(annotated, frame_idx,
                        track_ids_this_frame, boxes_this_frame)

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames  "
                  f"count={counter.count}", flush=True)

    cap.release()
    writer.release()
    summary_path = os.path.join(output_dir, "tracking_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "video_path": video_path,
            "output_video": out_path,
            "line_pt1": [int(v) for v in line_pt1],
            "line_pt2": [int(v) for v in line_pt2],
            "total_cross_count": int(counter.count),
            "crossed_ids": [int(tid) for tid in sorted(counter.crossed_ids)],
            "occlusion_events_saved": int(analyzer.saved_events),
            "event_cooldown_frames": int(event_cooldown),
        }, f, indent=2)
    print(f"\n[Done] Output video: {out_path}")
    print(f"[Done] Summary JSON: {summary_path}")
    print(f"[Done] Total objects crossed line: {counter.count}")
    print(f"[Done] Unique crossed IDs: {[int(tid) for tid in sorted(counter.crossed_ids)]}")
    print(f"[Done] Occlusion events saved: {analyzer.saved_events}")
    return counter.count


# ─── 6. CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2: Vehicle Detection & Tracking")
    sub = parser.add_subparsers(dest="cmd")

    # Fine-tune sub-command
    ft = sub.add_parser("finetune", help="Fine-tune YOLOv8 on vehicle dataset")
    ft.add_argument("--dataset_root", required=True)
    ft.add_argument("--model_size", default="yolov8n")
    ft.add_argument("--epochs", type=int, default=50)
    ft.add_argument("--batch",  type=int, default=16)
    ft.add_argument("--output_dir", default="./outputs/task2")

    # Track sub-command
    tr = sub.add_parser("track", help="Run MOT + cross-line counting on video")
    tr.add_argument("--model_path", required=True)
    tr.add_argument("--video_path", required=True)
    tr.add_argument("--output_dir", default="./outputs/task2")
    tr.add_argument("--conf",  type=float, default=0.35)
    tr.add_argument("--line_x1", type=int, default=0)
    tr.add_argument("--line_y1", type=int, default=0)
    tr.add_argument("--line_x2", type=int, default=0)
    tr.add_argument("--line_y2", type=int, default=0)
    tr.add_argument("--event_cooldown", type=int, default=30)

    args = parser.parse_args()

    if args.cmd == "finetune":
        yaml_path = create_dataset_yaml(args.dataset_root)
        finetune_yolo(yaml_path, args.model_size, args.epochs,
                      batch=args.batch, output_dir=args.output_dir)

    elif args.cmd == "track":
        run_tracking_pipeline(
            model_path=args.model_path,
            video_path=args.video_path,
            output_dir=args.output_dir,
            line_pt1=(args.line_x1, args.line_y1),
            line_pt2=(args.line_x2, args.line_y2),
            conf_thresh=args.conf,
            event_cooldown=args.event_cooldown,
        )
    else:
        parser.print_help()
