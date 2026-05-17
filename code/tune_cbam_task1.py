"""
Task 1 CBAM tuning experiments.

This script focuses only on CBAM variants for Flowers102 classification.
It tests insertion depth, CBAM strength, CBAM-specific learning rate, and
augmentation strength without changing the main Task 1 pipeline.

Example:
    python tune_cbam_task1.py --data_dir ../data/flowers102 --output_dir ../outputs/task1_cbam_tuning
"""

import argparse
import copy
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import models

import train as task1


def parse_layers(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def build_cbam_resnet(
    arch: str,
    num_classes: int,
    pretrained: bool,
    cbam_layers: list[str],
    cbam_reduction: int,
    cbam_kernel: int,
):
    weights = None
    if pretrained:
        if arch == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        elif arch == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        else:
            raise ValueError("CBAM tuning script supports resnet18/resnet34 only.")

    model = getattr(models, arch)(weights=weights)
    channel_map = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
    for layer_name in cbam_layers:
        if layer_name not in channel_map:
            raise ValueError(f"Unknown CBAM layer: {layer_name}")
        layer = getattr(model, layer_name)
        cbam = task1.CBAMBlock(
            channel_map[layer_name],
            reduction=cbam_reduction,
            kernel_size=cbam_kernel,
        )
        setattr(model, layer_name, nn.Sequential(layer, cbam))

    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, num_classes))
    return model


def get_cbam_param_groups(model, backbone_lr, cbam_lr, head_lr):
    head_params = list(model.fc.parameters())
    cbam_params = []
    cbam_ids = set()
    for module in model.modules():
        if isinstance(module, task1.CBAMBlock):
            for param in module.parameters():
                cbam_params.append(param)
                cbam_ids.add(id(param))

    head_ids = set(id(p) for p in head_params)
    backbone_params = [
        p for p in model.parameters()
        if id(p) not in head_ids and id(p) not in cbam_ids
    ]

    return [
        {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
        {"params": cbam_params, "lr": cbam_lr, "name": "cbam"},
        {"params": head_params, "lr": head_lr, "name": "head"},
    ]


def make_scheduler(optimizer, epochs, warmup_epochs):
    warmup_epochs = min(max(warmup_epochs, 0), max(epochs - 1, 0))
    if warmup_epochs > 0:
        return SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, epochs - warmup_epochs),
                    eta_min=1e-6,
                ),
            ],
            milestones=[warmup_epochs],
        )
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


def run_one(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = task1.get_dataloaders(
        cfg["data_dir"],
        batch_size=cfg["batch_size"],
    )
    model = build_cbam_resnet(
        arch=cfg["arch"],
        num_classes=102,
        pretrained=True,
        cbam_layers=cfg["cbam_layers"],
        cbam_reduction=cfg["cbam_reduction"],
        cbam_kernel=cfg["cbam_kernel"],
    ).to(device)

    optimizer = optim.AdamW(
        get_cbam_param_groups(
            model,
            backbone_lr=cfg["backbone_lr"],
            cbam_lr=cfg["cbam_lr"],
            head_lr=cfg["head_lr"],
        ),
        weight_decay=cfg["weight_decay"],
    )
    scheduler = make_scheduler(optimizer, cfg["epochs"], cfg["warmup_epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    best_val_acc = 0.0
    best_state = None
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = task1.train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, None, cfg
        )
        val_loss, val_acc = task1.evaluate(
            model, val_loader, criterion, device, epoch, "val", None
        )
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"{cfg['exp_name']} epoch {epoch:03d}/{cfg['epochs']} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

    model.load_state_dict(best_state)
    test_loss, test_acc = task1.evaluate(
        model, test_loader, criterion, device, cfg["epochs"], "test", None
    )

    os.makedirs(cfg["output_dir"], exist_ok=True)
    ckpt_path = Path(cfg["output_dir"]) / f"{cfg['exp_name']}_best.pth"
    torch.save({
        "config": cfg,
        "state_dict": best_state,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
    }, ckpt_path)
    hist_path = Path(cfg["output_dir"]) / f"{cfg['exp_name']}_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "exp_name": cfg["exp_name"],
        "arch": cfg["arch"],
        "cbam_layers": ",".join(cfg["cbam_layers"]),
        "cbam_reduction": cfg["cbam_reduction"],
        "cbam_kernel": cfg["cbam_kernel"],
        "backbone_lr": cfg["backbone_lr"],
        "cbam_lr": cfg["cbam_lr"],
        "head_lr": cfg["head_lr"],
        "mixup_alpha": cfg["mixup_alpha"],
        "cutmix_alpha": cfg["cutmix_alpha"],
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def preset_configs(args):
    base = dict(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        arch=args.arch,
        batch_size=args.batch_size,
        epochs=args.epochs,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
    )
    return [
        {
            **base,
            "exp_name": "cbam_l4_light",
            "cbam_layers": ["layer4"],
            "cbam_reduction": 32,
            "cbam_kernel": 3,
            "cbam_lr": 1e-3,
            "mixup_alpha": 0.0,
            "cutmix_alpha": 0.0,
        },
        {
            **base,
            "exp_name": "cbam_l34_mid",
            "cbam_layers": ["layer3", "layer4"],
            "cbam_reduction": 16,
            "cbam_kernel": 7,
            "cbam_lr": 5e-4,
            "mixup_alpha": 0.0,
            "cutmix_alpha": 0.0,
        },
        {
            **base,
            "exp_name": "cbam_l34_mixup",
            "cbam_layers": ["layer3", "layer4"],
            "cbam_reduction": 16,
            "cbam_kernel": 7,
            "cbam_lr": 5e-4,
            "mixup_alpha": 0.1,
            "cutmix_alpha": 0.0,
        },
        {
            **base,
            "exp_name": "cbam_all_baseline_lr",
            "cbam_layers": ["layer1", "layer2", "layer3", "layer4"],
            "cbam_reduction": 16,
            "cbam_kernel": 7,
            "cbam_lr": args.backbone_lr,
            "mixup_alpha": args.mixup_alpha,
            "cutmix_alpha": args.cutmix_alpha,
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Tune CBAM variants for Task 1.")
    parser.add_argument("--data_dir", default="../data/flowers102")
    parser.add_argument("--output_dir", default="../outputs/task1_cbam_tuning")
    parser.add_argument("--arch", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--cbam_lr", type=float, default=5e-4)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--single", action="store_true",
                        help="Run one custom config instead of presets")
    parser.add_argument("--exp_name", default="cbam_custom")
    parser.add_argument("--cbam_layers", default="layer3,layer4")
    parser.add_argument("--cbam_reduction", type=int, default=16)
    parser.add_argument("--cbam_kernel", type=int, default=7)
    args = parser.parse_args()

    if args.single:
        configs = [{
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "arch": args.arch,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "backbone_lr": args.backbone_lr,
            "cbam_lr": args.cbam_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "warmup_epochs": args.warmup_epochs,
            "mixup_alpha": args.mixup_alpha,
            "cutmix_alpha": args.cutmix_alpha,
            "exp_name": args.exp_name,
            "cbam_layers": parse_layers(args.cbam_layers),
            "cbam_reduction": args.cbam_reduction,
            "cbam_kernel": args.cbam_kernel,
        }]
    else:
        configs = preset_configs(args)

    rows = []
    for cfg in configs:
        print("\n" + "=" * 80)
        print(f"Running {cfg['exp_name']}")
        print("=" * 80)
        rows.append(run_one(cfg))

    summary_path = Path(args.output_dir) / "cbam_tuning_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            clean = row.copy()
            for key in ["best_val_acc", "test_loss", "test_acc"]:
                clean[key] = f"{float(clean[key]):.4f}"
            writer.writerow(clean)
    print(f"\n[Done] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
