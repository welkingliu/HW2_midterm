"""
Generate reusable visualizations from HW2 output files.

Examples:
    python visualize_results.py --outputs ../outputs
    python visualize_results.py --outputs ../outputs --video_frame
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_bar(labels, values, title, ylabel, out_path, color="#4E79A7",
             ylim=None, rotate=25):
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values, color=color)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def task1_visuals(outputs: Path, save_dir: Path):
    task_dir = outputs / "task1"
    summary = task_dir / "ablation_summary.csv"
    if not summary.exists():
        return

    rows = read_csv(summary)
    labels = [
        r["exp_name"].replace("resnet18_", "r18_").replace("resnet34_", "r34_")
        for r in rows
    ]
    save_bar(
        labels,
        [float(r["test_acc"]) for r in rows],
        "Task 1 Test Accuracy",
        "Test Accuracy",
        save_dir / "task1_test_accuracy.png",
        color="#4E79A7",
        ylim=(0, 1.0),
    )

    plt.figure(figsize=(8, 4.5))
    for history_path in sorted(task_dir.glob("*_history.json")):
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        name = history_path.stem.replace("_history", "")
        plt.plot(
            [item["epoch"] for item in history],
            [item["val_acc"] for item in history],
            label=name,
            linewidth=1.8,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Task 1 Validation Accuracy Curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_dir / "task1_val_accuracy_curves.png", dpi=200)
    plt.close()


def task2_visuals(outputs: Path, save_dir: Path, make_video_frame: bool):
    task_dir = outputs / "task2"
    yolo_csv = task_dir / "vehicle_finetune" / "results.csv"
    if yolo_csv.exists():
        df = pd.read_csv(yolo_csv)

        plt.figure(figsize=(7, 4.5))
        plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50", linewidth=2)
        plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.title("Task 2 YOLOv8 Validation mAP")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "task2_yolo_map_curve.png", dpi=200)
        plt.close()

        plt.figure(figsize=(7, 4.5))
        plt.plot(df["epoch"], df["train/box_loss"], label="train box")
        plt.plot(df["epoch"], df["train/cls_loss"], label="train cls")
        plt.plot(df["epoch"], df["val/box_loss"], label="val box")
        plt.plot(df["epoch"], df["val/cls_loss"], label="val cls")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Task 2 YOLOv8 Loss Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "task2_yolo_loss_curve.png", dpi=200)
        plt.close()

    summary_path = task_dir / "tracking_summary.json"
    if summary_path.exists():
        summary = json.load(open(summary_path, encoding="utf-8"))
        labels = ["Crossed IDs", "Occlusion Events"]
        values = [
            int(summary.get("total_cross_count", 0)),
            int(summary.get("occlusion_events_saved", 0)),
        ]
        save_bar(
            labels,
            values,
            "Task 2 Tracking Summary",
            "Count",
            save_dir / "task2_tracking_summary.png",
            color="#F28E2B",
            ylim=(0, max(values + [1]) + 3),
            rotate=0,
        )

    if make_video_frame:
        video = task_dir / "tracked_output.mp4"
        if video.exists():
            cap = cv2.VideoCapture(str(video))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count // 2, 0))
                ok, frame = cap.read()
                if ok:
                    cv2.imwrite(str(save_dir / "task2_tracked_midframe.jpg"), frame)
            cap.release()

    make_occlusion_montage(task_dir / "occlusion_frames", save_dir / "task2_occlusion_montage.jpg")


def make_occlusion_montage(occ_dir: Path, out_path: Path):
    if not occ_dir.exists():
        return
    images = []
    for event in sorted(occ_dir.glob("event_*"))[:4]:
        frames = sorted(event.glob("*.jpg"))
        if frames:
            images.append((event.name, frames[-1]))
    if not images:
        return

    thumbs = []
    for label, path in images:
        img = Image.open(path).convert("RGB")
        img.thumbnail((420, 280))
        canvas = Image.new("RGB", (420, 310), "white")
        canvas.paste(img, ((420 - img.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 286), label, fill=(0, 0, 0))
        thumbs.append(canvas)

    montage = Image.new("RGB", (840, 620), "white")
    for idx, thumb in enumerate(thumbs):
        x = (idx % 2) * 420
        y = (idx // 2) * 310
        montage.paste(thumb, (x, y))
    montage.save(out_path, quality=92)


def task3_visuals(outputs: Path, save_dir: Path):
    task_dir = outputs / "task3"
    summary = task_dir / "loss_comparison.csv"
    if not summary.exists():
        return

    rows = read_csv(summary)
    save_bar(
        [r["loss_mode"] for r in rows],
        [float(r["best_val_mIoU"]) for r in rows],
        "Task 3 Loss Function Comparison",
        "Best Val mIoU",
        save_dir / "task3_loss_comparison.png",
        color="#59A14F",
        ylim=(0, max(float(r["best_val_mIoU"]) for r in rows) + 0.08),
        rotate=0,
    )

    plt.figure(figsize=(8, 4.5))
    for history_path in sorted(task_dir.glob("history_*.json")):
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        mode = history_path.stem.replace("history_", "")
        plt.plot(
            [item["epoch"] for item in history],
            [item["val_mIoU"] for item in history],
            label=mode,
            linewidth=2,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation mIoU")
    plt.title("Task 3 Validation mIoU Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "task3_val_miou_curves.png", dpi=200)
    plt.close()

    per_class_files = sorted(task_dir.glob("per_class_iou_*.csv"))
    if per_class_files:
        plt.figure(figsize=(9, 4.8))
        for path in per_class_files:
            rows = read_csv(path)
            mode = path.stem.replace("per_class_iou_", "")
            plt.plot(
                [r["class_name"] for r in rows],
                [float(r["iou"]) for r in rows],
                marker="o",
                label=mode,
            )
        plt.ylim(0, 1.0)
        plt.ylabel("IoU")
        plt.title("Task 3 Per-class IoU")
        plt.xticks(rotation=25, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "task3_per_class_iou_all_losses.png", dpi=200)
        plt.close()


def write_summary_markdown(outputs: Path, save_dir: Path):
    lines = ["# HW2 Visualization Summary", ""]
    t1 = outputs / "task1" / "ablation_summary.csv"
    if t1.exists():
        rows = read_csv(t1)
        best = max(rows, key=lambda r: float(r["test_acc"]))
        lines.append(f"- Task 1 best test accuracy: {best['exp_name']} = {float(best['test_acc']):.4f}")
    t2 = outputs / "task2" / "tracking_summary.json"
    if t2.exists():
        s = json.load(open(t2, encoding="utf-8"))
        lines.append(f"- Task 2 crossed objects: {s.get('total_cross_count')}")
        lines.append(f"- Task 2 occlusion events: {s.get('occlusion_events_saved')}")
    t3 = outputs / "task3" / "loss_comparison.csv"
    if t3.exists():
        rows = read_csv(t3)
        best = max(rows, key=lambda r: float(r["best_val_mIoU"]))
        lines.append(f"- Task 3 best val mIoU: {best['loss_mode']} = {float(best['best_val_mIoU']):.4f}")
    (save_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate HW2 report/PPT visualizations.")
    parser.add_argument("--outputs", default="../outputs", help="Path to outputs directory")
    parser.add_argument("--save_dir", default="", help="Directory for generated figures")
    parser.add_argument("--video_frame", action="store_true", help="Export a middle frame from tracked_output.mp4")
    args = parser.parse_args()

    outputs = Path(args.outputs).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else outputs / "visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)

    task1_visuals(outputs, save_dir)
    task2_visuals(outputs, save_dir, args.video_frame)
    task3_visuals(outputs, save_dir)
    write_summary_markdown(outputs, save_dir)
    print(f"[Done] visualizations saved to: {save_dir}")


if __name__ == "__main__":
    main()
