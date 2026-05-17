"""
Unified launcher for HW2 tasks.

This script does not contain new model logic. It orchestrates the three task
scripts in this folder and keeps all outputs under one directory.
"""

import argparse
from pathlib import Path

import train as task1
import detect_track as task2
import unet_train as task3


def as_path(value: str) -> str:
    """Normalize CLI paths across Windows and Linux shells."""
    return str(Path(value.replace("\\", "/")).expanduser().resolve())


def parse_tasks(value: str) -> list[str]:
    if value == "all":
        return ["task1", "task2", "task3"]
    tasks = [item.strip().lower() for item in value.split(",") if item.strip()]
    valid = {"task1", "task2", "task3"}
    unknown = sorted(set(tasks) - valid)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown task(s): {', '.join(unknown)}")
    return tasks


def run_task1(args: argparse.Namespace) -> None:
    output_dir = Path(as_path(args.output_dir)) / "task1"
    if args.task1_mode == "ablation":
        task1.run_ablation(as_path(args.flowers_dir), str(output_dir))
        return

    cfg = {
        "data_dir": as_path(args.flowers_dir),
        "output_dir": str(output_dir),
        "mode": "single",
        "arch": args.task1_arch,
        "pretrained": args.task1_pretrained,
        "attention": args.task1_attention,
        "epochs": args.task1_epochs,
        "batch_size": args.task1_batch_size,
        "head_lr": args.task1_head_lr,
        "backbone_lr": args.task1_backbone_lr,
        "exp_name": args.task1_exp_name,
        "label_smoothing": args.task1_label_smoothing,
        "warmup_epochs": args.task1_warmup_epochs,
        "mixup_alpha": args.task1_mixup_alpha,
        "cutmix_alpha": args.task1_cutmix_alpha,
        "use_tracking": task1.SWANLAB_AVAILABLE and not args.disable_tracking,
    }
    task1.train(cfg)


def run_task2(args: argparse.Namespace) -> None:
    output_dir = Path(as_path(args.output_dir)) / "task2"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = as_path(args.task2_model_path) if args.task2_model_path else ""

    if args.task2_phase in {"finetune", "all"}:
        dataset_yaml = task2.create_dataset_yaml(
            as_path(args.vehicles_dir),
            output_path=str(output_dir / "vehicle_dataset.yaml"),
        )
        model_path = task2.finetune_yolo(
            dataset_yaml=dataset_yaml,
            model_size=args.task2_model_size,
            epochs=args.task2_epochs,
            imgsz=args.task2_imgsz,
            batch=args.task2_batch,
            output_dir=str(output_dir),
        )

    if args.task2_phase in {"track", "all"}:
        if not model_path:
            raise ValueError(
                "Task 2 tracking needs --task2_model_path, or use "
                "--task2_phase all to fine-tune first."
            )
        if not args.video_path:
            raise ValueError("Task 2 tracking needs --video_path.")

        task2.run_tracking_pipeline(
            model_path=model_path,
            video_path=as_path(args.video_path),
            output_dir=str(output_dir),
            line_pt1=(args.line_x1, args.line_y1),
            line_pt2=(args.line_x2, args.line_y2),
            conf_thresh=args.task2_conf,
            iou_thresh=args.task2_iou,
            event_cooldown=args.task2_event_cooldown,
        )


def run_task3(args: argparse.Namespace) -> None:
    output_dir = Path(as_path(args.output_dir)) / "task3"
    if args.task3_mode == "all":
        task3.run_comparison(
            data_dir=as_path(args.stanford_dir),
            output_dir=str(output_dir),
            epochs=args.task3_epochs,
            batch_size=args.task3_batch_size,
            lr=args.task3_lr,
            combined_alpha=args.task3_combined_alpha,
            class_weighted=args.task3_class_weighted,
            attention_unet=args.task3_attention_unet,
        )
        return

    cfg = {
        "data_dir": as_path(args.stanford_dir),
        "output_dir": str(output_dir),
        "loss_mode": args.task3_mode,
        "epochs": args.task3_epochs,
        "batch_size": args.task3_batch_size,
        "lr": args.task3_lr,
        "base_ch": args.task3_base_ch,
        "combined_alpha": args.task3_combined_alpha,
        "class_weighted": args.task3_class_weighted,
        "attention_unet": args.task3_attention_unet,
        "use_tracking": task3.SWANLAB_AVAILABLE and not args.disable_tracking,
    }
    task3.run_experiment(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HW2 Task 1, Task 2 and Task 3 from one entry point."
    )
    parser.add_argument(
        "--tasks",
        type=parse_tasks,
        default=["task1", "task2", "task3"],
        help="all, or comma-separated subset: task1,task2,task3",
    )
    parser.add_argument("--output_dir", default="../outputs")
    parser.add_argument("--disable_tracking", action="store_true")

    parser.add_argument("--flowers_dir", default="../data/flowers102")
    parser.add_argument("--task1_mode", choices=["single", "ablation"], default="ablation")
    parser.add_argument(
        "--task1_arch",
        choices=["resnet18", "resnet34", "efficientnet_b0", "convnext_tiny", "swin_t"],
        default="resnet18",
    )
    parser.add_argument("--task1_pretrained", action="store_true", default=True)
    parser.add_argument("--task1_no_pretrained", dest="task1_pretrained", action="store_false")
    parser.add_argument("--task1_attention", choices=["none", "se", "cbam"], default="none")
    parser.add_argument("--task1_epochs", type=int, default=30)
    parser.add_argument("--task1_batch_size", type=int, default=64)
    parser.add_argument("--task1_head_lr", type=float, default=1e-3)
    parser.add_argument("--task1_backbone_lr", type=float, default=1e-4)
    parser.add_argument("--task1_exp_name", default="resnet18_pretrained")
    parser.add_argument("--task1_label_smoothing", type=float, default=0.1)
    parser.add_argument("--task1_warmup_epochs", type=int, default=3)
    parser.add_argument("--task1_mixup_alpha", type=float, default=0.2)
    parser.add_argument("--task1_cutmix_alpha", type=float, default=1.0)

    parser.add_argument("--vehicles_dir", default="../data/road_vehicles")
    parser.add_argument("--video_path", default="")
    parser.add_argument("--task2_phase", choices=["finetune", "track", "all"], default="all")
    parser.add_argument("--task2_model_path", default="")
    parser.add_argument("--task2_model_size", default="yolov8n")
    parser.add_argument("--task2_epochs", type=int, default=50)
    parser.add_argument("--task2_imgsz", type=int, default=640)
    parser.add_argument("--task2_batch", type=int, default=16)
    parser.add_argument("--task2_conf", type=float, default=0.35)
    parser.add_argument("--task2_iou", type=float, default=0.45)
    parser.add_argument("--task2_event_cooldown", type=int, default=30)
    parser.add_argument("--line_x1", type=int, default=0)
    parser.add_argument("--line_y1", type=int, default=0)
    parser.add_argument("--line_x2", type=int, default=0)
    parser.add_argument("--line_y2", type=int, default=0)

    parser.add_argument("--stanford_dir", default="../data/StanfordBackground")
    parser.add_argument("--task3_mode", choices=["ce", "dice", "combined", "all"], default="all")
    parser.add_argument("--task3_epochs", type=int, default=40)
    parser.add_argument("--task3_batch_size", type=int, default=8)
    parser.add_argument("--task3_lr", type=float, default=1e-3)
    parser.add_argument("--task3_base_ch", type=int, default=32)
    parser.add_argument("--task3_combined_alpha", type=float, default=0.5)
    parser.add_argument("--task3_class_weighted", action="store_true")
    parser.add_argument("--task3_attention_unet", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if "task2" in args.tasks and args.task2_phase in {"track", "all"}:
        if not args.video_path:
            raise ValueError("Task 2 tracking needs --video_path.")

    if "task1" in args.tasks:
        run_task1(args)
    if "task2" in args.tasks:
        run_task2(args)
    if "task3" in args.tasks:
        run_task3(args)


if __name__ == "__main__":
    main()
