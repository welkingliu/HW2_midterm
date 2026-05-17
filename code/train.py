"""
Task 1: Fine-tuning a Pre-trained CNN on 102 Category Flower Dataset
Features:
- ResNet-18/34 with ImageNet pre-trained weights
- Differential learning rates (frozen backbone / new head)
- Pre-training ablation (pretrained vs random init)
- SE-block / CBAM attention integration
- SwanLab experiment tracking
"""

import os
import argparse
import time
import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# ─── Optional: SwanLab tracking (pip install swanlab) ───────────────────────
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("[INFO] swanlab not installed. Run: pip install swanlab  to enable tracking.")

# ─── 1. Attention Modules ────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


# ─── 2. ResNet with optional Attention ───────────────────────────────────────

def build_model(
    arch: str = "resnet18",
    num_classes: int = 102,
    pretrained: bool = True,
    attention: str = "none",   # "none" | "se" | "cbam"
) -> nn.Module:
    """
    Build ResNet model with:
    - Optional ImageNet pre-training
    - Modified FC head for num_classes
    - Optional attention blocks injected after each residual layer
    """
    weights = None
    if pretrained:
        if arch == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        elif arch == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        else:
            raise ValueError(f"Unsupported arch: {arch}")

    model = getattr(models, arch)(weights=weights)

    # ── Inject attention after each layer block ──────────────────────────────
    if attention != "none":
        channel_map = {
            "layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512
        }
        for layer_name, channels in channel_map.items():
            layer = getattr(model, layer_name)
            if attention == "se":
                att_block = SEBlock(channels)
            elif attention == "cbam":
                att_block = CBAMBlock(channels)
            else:
                att_block = None

            if att_block is not None:
                setattr(model, layer_name, nn.Sequential(layer, att_block))

    # ── Replace final FC ──────────────────────────────────────────────────────
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )

    return model


def get_param_groups(model: nn.Module, head_lr: float, backbone_lr: float):
    """Differential learning rates: backbone gets backbone_lr, head gets head_lr."""
    head_params = list(model.fc.parameters())
    head_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ]


# ─── 3. Data Loading ──────────────────────────────────────────────────────────

def build_model(
    arch: str = "resnet18",
    num_classes: int = 102,
    pretrained: bool = True,
    attention: str = "none",
) -> nn.Module:
    """Improved builder: ResNet plus EfficientNet/ConvNeXt/Swin backbones."""
    weights = None
    if pretrained:
        weights_map = {
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
            "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
            "convnext_tiny": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
            "swin_t": models.Swin_T_Weights.IMAGENET1K_V1,
        }
        if arch not in weights_map:
            raise ValueError(f"Unsupported arch: {arch}")
        weights = weights_map[arch]

    model = getattr(models, arch)(weights=weights)

    if attention != "none":
        if not arch.startswith("resnet"):
            print(f"[WARN] attention={attention} is only injected for ResNet; ignored for {arch}.")
        else:
            for layer_name, channels in {
                "layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512
            }.items():
                layer = getattr(model, layer_name)
                att_block = SEBlock(channels) if attention == "se" else CBAMBlock(channels)
                setattr(model, layer_name, nn.Sequential(layer, att_block))

    if arch.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, num_classes))
    elif arch in {"efficientnet_b0", "convnext_tiny"}:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch == "swin_t":
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return model


def get_param_groups(model: nn.Module, head_lr: float, backbone_lr: float):
    """Differential LR for common torchvision classification heads."""
    if hasattr(model, "fc"):
        head_params = list(model.fc.parameters())
    elif hasattr(model, "classifier"):
        head_params = list(model.classifier.parameters())
    elif hasattr(model, "head"):
        head_params = list(model.head.parameters())
    else:
        raise AttributeError("Could not locate classification head.")
    head_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]


def get_dataloaders(data_dir: str, batch_size: int = 64, val_split: float = 0.15):
    """
    Oxford 102 Flowers dataset.
    Expected structure:  data_dir/  (one sub-folder per class, e.g. class_001/ … class_102/)
    If using torchvision's Flowers102 download, use split='train'/'val'/'test'.
    Falls back to ImageFolder if the downloaded structure is not present.
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Try torchvision Flowers102 first
    try:
        from torchvision.datasets import Flowers102
        train_ds = Flowers102(data_dir, split="train", transform=train_tf, download=True)
        val_ds   = Flowers102(data_dir, split="val",   transform=val_tf,   download=True)
        test_ds  = Flowers102(data_dir, split="test",  transform=val_tf,   download=True)
        print(f"[Data] Flowers102  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    except Exception:
        # Fallback: ImageFolder
        full_ds = datasets.ImageFolder(data_dir, transform=train_tf)
        n_val = int(len(full_ds) * val_split)
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        val_ds.dataset.transform = val_tf
        test_ds = val_ds
        print(f"[Data] ImageFolder  train={n_train}  val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


def mixup_or_cutmix(imgs, labels, num_classes, mixup_alpha=0.0, cutmix_alpha=0.0):
    """Apply one batch-level regularization method and return soft labels."""
    labels_onehot = torch.zeros(imgs.size(0), num_classes, device=imgs.device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)

    use_mixup = mixup_alpha > 0
    use_cutmix = cutmix_alpha > 0
    if not use_mixup and not use_cutmix:
        return imgs, labels_onehot

    perm = torch.randperm(imgs.size(0), device=imgs.device)
    if use_cutmix and (not use_mixup or random.random() < 0.5):
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        _, _, h, w = imgs.shape
        cut_ratio = (1.0 - lam) ** 0.5
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, w)
        y2 = min(cy + cut_h // 2, h)
        imgs = imgs.clone()
        imgs[:, :, y1:y2, x1:x2] = imgs[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(w * h))
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        imgs = lam * imgs + (1.0 - lam) * imgs[perm]

    soft_labels = lam * labels_onehot + (1.0 - lam) * labels_onehot[perm]
    return imgs, soft_labels


def soft_cross_entropy(logits, soft_targets, label_smoothing=0.0):
    if label_smoothing > 0:
        num_classes = soft_targets.size(1)
        soft_targets = soft_targets * (1.0 - label_smoothing) + label_smoothing / num_classes
    log_probs = torch.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


# ─── 4. Training Loop ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, run=None, cfg=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        original_labels = labels
        mixup_alpha = (cfg or {}).get("mixup_alpha", 0.0)
        cutmix_alpha = (cfg or {}).get("cutmix_alpha", 0.0)
        use_soft_targets = mixup_alpha > 0 or cutmix_alpha > 0
        if use_soft_targets:
            imgs, labels = mixup_or_cutmix(
                imgs, labels, num_classes=102,
                mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
            )
        optimizer.zero_grad()
        outputs = model(imgs)
        if use_soft_targets:
            loss = soft_cross_entropy(
                outputs, labels,
                label_smoothing=(cfg or {}).get("label_smoothing", 0.1),
            )
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(original_labels).sum().item()
        total += imgs.size(0)

        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}]  "
                  f"Loss={loss.item():.4f}", flush=True)

    avg_loss = total_loss / total
    acc = correct / total
    if run:
        run.log({"train/loss": avg_loss, "train/acc": acc, "epoch": epoch})
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch=0, split="val", run=None):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    if run:
        run.log({f"{split}/loss": avg_loss, f"{split}/acc": acc, "epoch": epoch})
    return avg_loss, acc


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── SwanLab init ─────────────────────────────────────────────────────────
    run = None
    if SWANLAB_AVAILABLE and cfg.get("use_tracking", True):
        run = swanlab.init(
            project="flower102-classification",
            experiment_name=cfg.get("exp_name", "baseline"),
            config=cfg,
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg["data_dir"], batch_size=cfg["batch_size"]
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_model(
        arch=cfg["arch"],
        num_classes=102,
        pretrained=cfg["pretrained"],
        attention=cfg.get("attention", "none"),
    ).to(device)
    print(f"[Model] {cfg['arch']}  pretrained={cfg['pretrained']}  "
          f"attention={cfg.get('attention','none')}")

    # ── Optimizer with differential LR ───────────────────────────────────────
    param_groups = get_param_groups(
        model,
        head_lr=cfg["head_lr"],
        backbone_lr=cfg["backbone_lr"],
    )
    optimizer = optim.AdamW(param_groups, weight_decay=cfg.get("weight_decay", 1e-4))
    warmup_epochs = int(cfg.get("warmup_epochs", 3))
    cosine_epochs = max(1, cfg["epochs"] - warmup_epochs)
    if warmup_epochs > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.1))

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, run, cfg
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, epoch, "val", run
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{cfg['epochs']}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"({elapsed:.1f}s)")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device,
                                   cfg["epochs"], "test", run)
    print(f"\n[Result] Best val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(cfg["output_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["output_dir"], f"{cfg['exp_name']}_best.pth")
    torch.save({"config": cfg, "state_dict": best_state,
                "best_val_acc": best_val_acc, "test_acc": test_acc}, ckpt_path)

    hist_path = os.path.join(cfg["output_dir"], f"{cfg['exp_name']}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    if run:
        run.finish()

    return best_val_acc, test_acc


# ─── 5. Ablation Runner ───────────────────────────────────────────────────────

def run_ablation(data_dir: str, output_dir: str):
    """
    Runs all ablation experiments and writes a summary CSV.
    Experiments:
      1. ResNet-18 pretrained  (baseline)
      2. ResNet-18 random init (ablation: no pretraining)
      3. ResNet-18 pretrained + SE attention
      4. ResNet-18 pretrained + CBAM attention
      5. ResNet-34 pretrained  (larger backbone)
    """
    import csv

    base_cfg = dict(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=64,
        epochs=30,
        head_lr=1e-3,
        backbone_lr=1e-4,
        weight_decay=1e-4,
        label_smoothing=0.1,
        warmup_epochs=3,
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        use_tracking=SWANLAB_AVAILABLE,
    )

    experiments = [
        {**base_cfg, "exp_name": "resnet18_pretrained",      "arch": "resnet18", "pretrained": True,  "attention": "none"},
        {**base_cfg, "exp_name": "resnet18_random",          "arch": "resnet18", "pretrained": False, "attention": "none"},
        {**base_cfg, "exp_name": "resnet18_pretrained_se",   "arch": "resnet18", "pretrained": True,  "attention": "se"},
        {**base_cfg, "exp_name": "resnet18_pretrained_cbam", "arch": "resnet18", "pretrained": True,  "attention": "cbam"},
        {**base_cfg, "exp_name": "resnet34_pretrained",      "arch": "resnet34", "pretrained": True,  "attention": "none"},
    ]

    results = []
    for cfg in experiments:
        print(f"\n{'='*60}")
        print(f"  Experiment: {cfg['exp_name']}")
        print(f"{'='*60}")
        val_acc, test_acc = train(cfg)
        results.append({
            "exp_name":  cfg["exp_name"],
            "arch":      cfg["arch"],
            "pretrained": cfg["pretrained"],
            "attention": cfg["attention"],
            "best_val_acc": f"{val_acc:.4f}",
            "test_acc":     f"{test_acc:.4f}",
        })

    summary_path = os.path.join(output_dir, "ablation_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[Ablation Summary saved to {summary_path}]")
    for r in results:
        print(f"  {r['exp_name']:40s}  val={r['best_val_acc']}  test={r['test_acc']}")


# ─── 6. CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1: Flower102 Classification")
    parser.add_argument("--data_dir",   default="./data/flowers102")
    parser.add_argument("--output_dir", default="./outputs/task1")
    parser.add_argument("--mode",       choices=["single", "ablation"], default="single")
    # single-run options
    parser.add_argument(
        "--arch",
        choices=["resnet18", "resnet34", "efficientnet_b0", "convnext_tiny", "swin_t"],
        default="resnet18",
    )
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--attention",  choices=["none", "se", "cbam"], default="none")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--head_lr",    type=float, default=1e-3)
    parser.add_argument("--backbone_lr",type=float, default=1e-4)
    parser.add_argument("--exp_name",   default="experiment")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    args = parser.parse_args()

    if args.mode == "ablation":
        run_ablation(args.data_dir, args.output_dir)
    else:
        cfg = vars(args)
        cfg["use_tracking"] = SWANLAB_AVAILABLE
        train(cfg)
