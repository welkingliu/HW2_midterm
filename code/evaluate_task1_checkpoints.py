"""
Evaluate Task 1 checkpoints on the Flowers102 test split.

This script is useful when extra single-run models exist in outputs/task1
but ablation_summary.csv does not contain their test accuracy.

Example:
    python evaluate_task1_checkpoints.py --data_dir ../data/flowers102 --output_dir ../outputs/task1
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

import train as task1


def infer_config_from_name(exp_name: str) -> dict:
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


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def evaluate_checkpoint(ckpt_path: Path, data_dir: str, batch_size: int):
    ckpt = load_checkpoint(ckpt_path)
    exp_name = ckpt_path.name.replace("_best.pth", "")
    cfg = ckpt.get("config", infer_config_from_name(exp_name))
    cfg = {**infer_config_from_name(exp_name), **cfg}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = task1.get_dataloaders(data_dir, batch_size=batch_size)

    model = task1.build_model(
        arch=cfg["arch"],
        num_classes=102,
        pretrained=False,
        attention=cfg.get("attention", "none"),
    )
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.1))
    test_loss, test_acc = task1.evaluate(
        model, test_loader, criterion, device, epoch=0, split="test", run=None
    )
    return {
        "exp_name": exp_name,
        "arch": cfg["arch"],
        "pretrained": cfg.get("pretrained", True),
        "attention": cfg.get("attention", "none"),
        "best_val_acc": ckpt.get("best_val_acc", ""),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "checkpoint": str(ckpt_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all Task 1 best checkpoints.")
    parser.add_argument("--data_dir", default="../data/flowers102")
    parser.add_argument("--output_dir", default="../outputs/task1")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pattern", default="*_best.pth")
    parser.add_argument("--summary_name", default="task1_full_summary.csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    ckpts = sorted(output_dir.glob(args.pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched {output_dir / args.pattern}")

    rows = []
    for ckpt_path in ckpts:
        print(f"\n[Eval] {ckpt_path.name}")
        row = evaluate_checkpoint(ckpt_path, args.data_dir, args.batch_size)
        print(f"  test_acc={row['test_acc']:.4f}  test_loss={row['test_loss']:.4f}")
        rows.append(row)

    summary_path = output_dir / args.summary_name
    fieldnames = [
        "exp_name", "arch", "pretrained", "attention",
        "best_val_acc", "test_loss", "test_acc", "checkpoint",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = row.copy()
            for key in ["best_val_acc", "test_loss", "test_acc"]:
                if clean[key] != "":
                    clean[key] = f"{float(clean[key]):.4f}"
            writer.writerow(clean)

    json_path = output_dir / args.summary_name.replace(".csv", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\n[Done] CSV saved to {summary_path}")
    print(f"[Done] JSON saved to {json_path}")


if __name__ == "__main__":
    main()
