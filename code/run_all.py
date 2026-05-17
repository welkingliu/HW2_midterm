"""
One-shot runner for all HW2 experiments.

It runs:
1. Task 1 full suite:
   - the original ablation suite
   - extra pretrained backbones: EfficientNet-B0, ConvNeXt-Tiny, Swin-T
2. Task 2:
   - YOLOv8 fine-tuning + tracking/counting if a video is provided
3. Task 3:
   - CE / Dice / Combined baseline comparison
   - optional improved Attention U-Net + class-weighted comparison
4. Visualization generation from all outputs.

Example:
    python run_all.py --video_path ../data/test_video.mp4
"""

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def as_path(value: str) -> str:
    return str(Path(value.replace("\\", "/")).expanduser().resolve())


def run(cmd: list[str], dry_run: bool = False):
    print("\n" + "=" * 80)
    print(" ".join(cmd))
    print("=" * 80, flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=HERE, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run all HW2 tasks end-to-end.")
    parser.add_argument("--flowers_dir", default="../data/flowers102")
    parser.add_argument("--vehicles_dir", default="../data/road_vehicles")
    parser.add_argument("--stanford_dir", default="../data/stanfordBackground")
    parser.add_argument("--video_path", default="", help="Required for Task 2 tracking")
    parser.add_argument("--output_dir", default="../outputs")
    parser.add_argument("--task1_epochs", type=int, default=30)
    parser.add_argument("--task2_epochs", type=int, default=50)
    parser.add_argument("--task2_model_size", default="yolov8n")
    parser.add_argument("--task3_epochs", type=int, default=40)
    parser.add_argument("--task3_batch_size", type=int, default=8)
    parser.add_argument("--task3_combined_alpha", type=float, default=0.5)
    parser.add_argument("--skip_task1", action="store_true")
    parser.add_argument("--skip_task2", action="store_true")
    parser.add_argument("--skip_task3", action="store_true")
    parser.add_argument("--skip_improved_task3", action="store_true")
    parser.add_argument("--skip_visualize", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    output_dir = as_path(args.output_dir)

    if not args.skip_task1:
        run([
            py, "main.py",
            "--tasks", "task1",
            "--flowers_dir", as_path(args.flowers_dir),
            "--output_dir", output_dir,
            "--task1_mode", "ablation",
        ], args.dry_run)

        for arch in ["efficientnet_b0", "convnext_tiny", "swin_t"]:
            run([
                py, "main.py",
                "--tasks", "task1",
                "--flowers_dir", as_path(args.flowers_dir),
                "--output_dir", output_dir,
                "--task1_mode", "single",
                "--task1_arch", arch,
                "--task1_epochs", str(args.task1_epochs),
                "--task1_exp_name", f"{arch}_pretrained",
            ], args.dry_run)

    if not args.skip_task2:
        task2_phase = "all" if args.video_path else "finetune"
        cmd = [
            py, "main.py",
            "--tasks", "task2",
            "--vehicles_dir", as_path(args.vehicles_dir),
            "--output_dir", output_dir,
            "--task2_phase", task2_phase,
            "--task2_model_size", args.task2_model_size,
            "--task2_epochs", str(args.task2_epochs),
        ]
        if args.video_path:
            cmd.extend(["--video_path", as_path(args.video_path)])
        run(cmd, args.dry_run)

    if not args.skip_task3:
        run([
            py, "main.py",
            "--tasks", "task3",
            "--stanford_dir", as_path(args.stanford_dir),
            "--output_dir", output_dir,
            "--task3_mode", "all",
            "--task3_epochs", str(args.task3_epochs),
            "--task3_batch_size", str(args.task3_batch_size),
            "--task3_combined_alpha", str(args.task3_combined_alpha),
        ], args.dry_run)

        if not args.skip_improved_task3:
            run([
                py, "main.py",
                "--tasks", "task3",
                "--stanford_dir", as_path(args.stanford_dir),
                "--output_dir", output_dir,
                "--task3_mode", "combined",
                "--task3_epochs", str(args.task3_epochs),
                "--task3_batch_size", str(args.task3_batch_size),
                "--task3_combined_alpha", str(args.task3_combined_alpha),
                "--task3_class_weighted",
                "--task3_attention_unet",
            ], args.dry_run)

    if not args.skip_visualize:
        viz_cmd = [
            py, "visualize_results.py",
            "--outputs", output_dir,
        ]
        if args.video_path:
            viz_cmd.append("--video_frame")
        run(viz_cmd, args.dry_run)

    print("\n[Done] All requested HW2 experiments finished.")


if __name__ == "__main__":
    main()
