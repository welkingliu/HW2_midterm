"""
Replay and summarize all HW2 results from saved outputs and weights.

Default behavior is lightweight: read existing CSV/JSON results and regenerate
visualizations. Optional flags load saved weights to recompute metrics.

Examples:
    # Summarize existing results and regenerate visualizations.
    python replay_all_results.py --outputs ../outputs

    # Re-evaluate Task 1 checkpoints on Flowers102 test split.
    python replay_all_results.py --outputs ../outputs --recompute_task1 --flowers_dir ../data/flowers102

    # Re-evaluate Task 3 U-Net checkpoints on Stanford validation split.
    python replay_all_results.py --outputs ../outputs --recompute_task3 --stanford_dir ../data/stanfordBackground

    # Re-run Task 2 tracking from saved YOLO weights.
    python replay_all_results.py --outputs ../outputs --rerun_task2 --video_path ../data/test_video.mp4

    # Full replay.
    python replay_all_results.py --outputs ../outputs --flowers_dir ../data/flowers102 \
        --stanford_dir ../data/stanfordBackground --video_path ../data/test_video.mp4 \
        --recompute_task1 --recompute_task3 --rerun_task2
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

import detect_track
import train as task1
import unet_train as task3

try:
    import tune_cbam_task1
except Exception:
    tune_cbam_task1 = None


HERE = Path(__file__).resolve().parent


def as_path(value: str) -> str:
    return str(Path(value.replace("\\", "/")).expanduser().resolve())


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_task1_config(exp_name: str):
    arch = exp_name
    pretrained = True
    attention = "none"
    if exp_name == "resnet18_random":
        arch = "resnet18"
        pretrained = False
    elif exp_name.endswith("_pretrained_se"):
        arch = exp_name.replace("_pretrained_se", "")
        attention = "se"
    elif exp_name.endswith("_pretrained_cbam"):
        arch = exp_name.replace("_pretrained_cbam", "")
        attention = "cbam"
    elif exp_name.endswith("_pretrained"):
        arch = exp_name.replace("_pretrained", "")
    return {
        "exp_name": exp_name,
        "arch": arch,
        "pretrained": pretrained,
        "attention": attention,
    }


def build_task1_model_from_ckpt(ckpt: dict, exp_name: str):
    cfg = {**infer_task1_config(exp_name), **ckpt.get("config", {})}
    if "cbam_layers" in cfg:
        if tune_cbam_task1 is None:
            raise RuntimeError("tune_cbam_task1.py is required for CBAM tuning checkpoints.")
        model = tune_cbam_task1.build_cbam_resnet(
            arch=cfg.get("arch", "resnet18"),
            num_classes=102,
            pretrained=False,
            cbam_layers=cfg["cbam_layers"],
            cbam_reduction=cfg.get("cbam_reduction", 16),
            cbam_kernel=cfg.get("cbam_kernel", 7),
        )
    else:
        model = task1.build_model(
            arch=cfg["arch"],
            num_classes=102,
            pretrained=False,
            attention=cfg.get("attention", "none"),
        )
    return model, cfg


def recompute_task1(outputs: Path, weights_dir: Path, flowers_dir: str, batch_size: int):
    task_weights = weights_dir / "task1"
    ckpts = sorted(task_weights.glob("*_best.pth"))
    if not ckpts:
        print(f"[Task1] No checkpoints found under {task_weights}")
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = task1.get_dataloaders(flowers_dir, batch_size=batch_size)
    rows = []
    for path in ckpts:
        exp_name = path.name.replace("_best.pth", "")
        print(f"[Task1] Evaluating {path.name}")
        ckpt = torch_load(path)
        model, cfg = build_task1_model_from_ckpt(ckpt, exp_name)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        model.to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.1))
        test_loss, test_acc = task1.evaluate(
            model, test_loader, criterion, device, epoch=0, split="test", run=None
        )
        rows.append({
            "exp_name": exp_name,
            "arch": cfg.get("arch", ""),
            "pretrained": cfg.get("pretrained", True),
            "attention": cfg.get("attention", "cbam_tuned" if "cbam_layers" in cfg else "none"),
            "best_val_acc": f"{float(ckpt.get('best_val_acc', 0.0)):.4f}",
            "test_loss": f"{test_loss:.4f}",
            "test_acc": f"{test_acc:.4f}",
            "checkpoint": str(path),
        })

    out_path = outputs / "replay" / "task1_recomputed_summary.csv"
    write_csv(out_path, rows)
    print(f"[Task1] Recomputed summary saved to {out_path}")
    return rows


def recompute_task3(outputs: Path, weights_dir: Path, stanford_dir: str, batch_size: int):
    task_weights = weights_dir / "task3"
    ckpts = sorted(task_weights.glob("unet_*_best.pth"))
    if not ckpts:
        print(f"[Task3] No checkpoints found under {task_weights}")
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = task3.get_dataloaders(stanford_dir, batch_size=batch_size)
    rows = []
    for path in ckpts:
        print(f"[Task3] Evaluating {path.name}")
        ckpt = torch_load(path)
        cfg = ckpt.get("config", {})
        loss_mode = cfg.get("loss_mode", path.stem.replace("unet_", "").replace("_best", ""))
        num_classes = task3.StanfordBackgroundDataset.NUM_CLASSES
        model = task3.UNet(
            in_channels=3,
            num_classes=num_classes,
            base_ch=cfg.get("base_ch", 32),
            use_attention=cfg.get("attention_unet", False),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        criterion = task3.get_loss_fn(
            loss_mode,
            num_classes,
            combined_alpha=cfg.get("combined_alpha", 0.5),
        )
        val_loss, val_miou, per_class = task3.evaluate(
            model,
            val_loader,
            criterion,
            device,
            epoch=0,
            num_classes=num_classes,
            run=None,
            split="val",
            return_per_class=True,
        )
        row = {
            "loss_mode": loss_mode,
            "val_loss": f"{val_loss:.4f}",
            "val_mIoU": f"{val_miou:.4f}",
            "checkpoint_best_val_mIoU": f"{float(ckpt.get('best_val_mIoU', 0.0)):.4f}",
            "checkpoint": str(path),
        }
        for i, value in enumerate(per_class):
            name = task3.StanfordBackgroundDataset.CLASS_NAMES[i]
            row[f"iou_{name}"] = "" if value != value else f"{float(value):.4f}"
        rows.append(row)

    out_path = outputs / "replay" / "task3_recomputed_summary.csv"
    write_csv(out_path, rows)
    print(f"[Task3] Recomputed summary saved to {out_path}")
    return rows


def rerun_task2(outputs: Path, weights_dir: Path, video_path: str):
    model_path = weights_dir / "task2" / "weights" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Task 2 best.pt not found: {model_path}")
    replay_dir = outputs / "replay" / "task2_tracking"
    replay_dir.mkdir(parents=True, exist_ok=True)
    count = detect_track.run_tracking_pipeline(
        model_path=str(model_path),
        video_path=video_path,
        output_dir=str(replay_dir),
    )
    print(f"[Task2] Replayed tracking count: {count}")


def summarize_existing(outputs: Path, weights_dir: Path):
    replay_dir = outputs / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# HW2 Replay Summary", ""]

    task1_summary = outputs / "task1" / "task1_full_summary.csv"
    if task1_summary.exists():
        rows = read_csv(task1_summary)
        if rows:
            best = max(rows, key=lambda r: float(r["test_acc"]))
            lines.append(f"- Task 1 best test accuracy: {best['exp_name']} = {float(best['test_acc']):.4f}")

    cbam_summary = outputs / "task1_cbam_tuning" / "cbam_tuning_summary.csv"
    if cbam_summary.exists():
        rows = read_csv(cbam_summary)
        if rows:
            best = max(rows, key=lambda r: float(r["test_acc"]))
            lines.append(f"- Task 1 best tuned CBAM: {best['exp_name']} = {float(best['test_acc']):.4f}")

    task2_summary = outputs / "task2" / "tracking_summary.json"
    if task2_summary.exists():
        s = json.load(open(task2_summary, encoding="utf-8"))
        lines.append(f"- Task 2 crossed objects: {s.get('total_cross_count')}")
        lines.append(f"- Task 2 occlusion events: {s.get('occlusion_events_saved')}")

    task3_summary = outputs / "task3" / "loss_comparison.csv"
    if task3_summary.exists():
        rows = read_csv(task3_summary)
        if rows:
            best = max(rows, key=lambda r: float(r["best_val_mIoU"]))
            lines.append(f"- Task 3 best val mIoU: {best['loss_mode']} = {float(best['best_val_mIoU']):.4f}")

    lines.append("")
    lines.append("## Available weights")
    for path in sorted(weights_dir.rglob("*.pt")) + sorted(weights_dir.rglob("*.pth")):
        lines.append(f"- `{path.relative_to(weights_dir)}`")

    out_path = replay_dir / "summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Summary] Existing result summary saved to {out_path}")


def regenerate_visualizations(outputs: Path, video_frame: bool):
    cmd = [
        sys.executable,
        str(HERE / "visualize_results.py"),
        "--outputs",
        str(outputs),
    ]
    if video_frame:
        cmd.append("--video_frame")
    subprocess.run(cmd, cwd=HERE, check=True)


def main():
    parser = argparse.ArgumentParser(description="Replay/summarize all HW2 saved results.")
    parser.add_argument("--outputs", default="../outputs")
    parser.add_argument("--weights_dir", default="", help="Default: <outputs>/train")
    parser.add_argument("--flowers_dir", default="../data/flowers102")
    parser.add_argument("--stanford_dir", default="../data/stanfordBackground")
    parser.add_argument("--video_path", default="")
    parser.add_argument("--batch_size_task1", type=int, default=64)
    parser.add_argument("--batch_size_task3", type=int, default=8)
    parser.add_argument("--recompute_task1", action="store_true")
    parser.add_argument("--recompute_task3", action="store_true")
    parser.add_argument("--rerun_task2", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Regenerate visualization figures")
    parser.add_argument("--video_frame", action="store_true", help="Also export a tracked video frame")
    args = parser.parse_args()

    outputs = Path(as_path(args.outputs))
    weights_dir = Path(as_path(args.weights_dir)) if args.weights_dir else outputs / "train"

    summarize_existing(outputs, weights_dir)

    if args.recompute_task1:
        recompute_task1(
            outputs,
            weights_dir,
            as_path(args.flowers_dir),
            args.batch_size_task1,
        )
    if args.recompute_task3:
        recompute_task3(
            outputs,
            weights_dir,
            as_path(args.stanford_dir),
            args.batch_size_task3,
        )
    if args.rerun_task2:
        if not args.video_path:
            raise ValueError("--rerun_task2 requires --video_path")
        rerun_task2(outputs, weights_dir, as_path(args.video_path))
    if args.visualize:
        regenerate_visualizations(outputs, args.video_frame)

    print("[Done] replay_all_results finished.")


if __name__ == "__main__":
    main()
