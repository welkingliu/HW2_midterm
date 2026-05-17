"""
Task 3: U-Net from Scratch + Loss Function Engineering
Features:
- Full U-Net (encoder / decoder / skip connections) — NO pretrained weights
- Manually coded Dice Loss
- Three training configs: CE only, Dice only, CE + Dice combined
- mIoU comparison on Stanford Background Dataset
"""

import os
import json
import time
import argparse
import random
import copy
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
#  Optional SwanLab tracking
# ─────────────────────────────────────────────────────────────────────────────
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False


# ─── 1. Building Blocks ───────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two × (Conv3x3 → BN → ReLU)"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock → save skip → MaxPool2×2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)      # feature map to be skip-connected
        down = self.pool(skip)   # downsampled for next encoder stage
        return skip, down


class AttentionGate(nn.Module):
    """Attention U-Net gate for filtering encoder skip features."""
    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.gate_proj = nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False)
        self.skip_proj = nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode="bilinear",
                                 align_corners=False)
        att = self.psi(self.gate_proj(gate) + self.skip_proj(skip))
        return skip * att


class DecoderBlock(nn.Module):
    """TransposedConv upsample → concat skip → ConvBlock"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 use_attention: bool = False):
        super().__init__()
        up_ch = in_ch // 2
        self.up = nn.ConvTranspose2d(in_ch, up_ch, kernel_size=2, stride=2)
        self.att_gate = AttentionGate(up_ch, skip_ch, max(out_ch // 2, 1)) \
            if use_attention else None
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd spatial dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        if self.att_gate is not None:
            skip = self.att_gate(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─── 2. U-Net ─────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Classic U-Net with 4 encoder levels + bottleneck + 4 decoder levels.
    All weights are randomly initialised — no pretrained backbone.

    channels: [64, 128, 256, 512]  (standard U-Net widths, halved for speed)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 8,
                 base_ch: int = 64, use_attention: bool = False):
        super().__init__()
        c = base_ch

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = EncoderBlock(in_channels, c)       # skip: c
        self.enc2 = EncoderBlock(c,           c*2)     # skip: c*2
        self.enc3 = EncoderBlock(c*2,         c*4)     # skip: c*4
        self.enc4 = EncoderBlock(c*4,         c*8)     # skip: c*8

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(c*8, c*16)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec4 = DecoderBlock(c*16, c*8,  c*8, use_attention)
        self.dec3 = DecoderBlock(c*8,  c*4,  c*4, use_attention)
        self.dec2 = DecoderBlock(c*4,  c*2,  c*2, use_attention)
        self.dec1 = DecoderBlock(c*2,  c,    c,   use_attention)

        # ── Output head ───────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(c, num_classes, kernel_size=1)

        # ── Random weight init (explicit, no pretrained) ──────────────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.out_conv(x)      # [B, num_classes, H, W]


# ─── 3. Loss Functions ────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss (soft version).
    Addresses foreground/background pixel imbalance.

    dice_loss = 1 - (2 * |X ∩ Y| + ε) / (|X| + |Y| + ε)
    averaged over all classes, then over the batch.
    """
    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, C, H, W]  (raw scores, not softmaxed)
        targets: [B, H, W]     (integer class indices)
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)             # [B, C, H, W]

        # One-hot encode targets → [B, C, H, W]
        B, H, W = targets.shape
        targets_onehot = torch.zeros_like(probs)
        valid_mask = (targets != self.ignore_index)
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0
        targets_onehot.scatter_(1, safe_targets.unsqueeze(1), 1.0)

        # Mask out ignore pixels
        if self.ignore_index >= 0:
            mask = valid_mask.unsqueeze(1).float()   # [B, 1, H, W]
            probs = probs * mask
            targets_onehot = targets_onehot * mask

        # Dice per class
        dims = (0, 2, 3)   # sum over batch, H, W
        intersection = (probs * targets_onehot).sum(dim=dims)  # [C]
        sum_preds    = probs.sum(dim=dims)
        sum_targets  = targets_onehot.sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / \
                         (sum_preds + sum_targets + self.smooth)
        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """CE Loss + Dice Loss with configurable weight λ."""
    def __init__(self, alpha: float = 0.5, ignore_index: int = 255,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.ce   = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                        weight=class_weights)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.ce(logits, targets) + \
               (1 - self.alpha) * self.dice(logits, targets)


def get_loss_fn(
    mode: str,
    num_classes: int,
    ignore_index: int = 255,
    combined_alpha: float = 0.5,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    if mode == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index,
                                   weight=class_weights)
    elif mode == "dice":
        return DiceLoss(ignore_index=ignore_index)
    elif mode == "combined":
        return CombinedLoss(alpha=combined_alpha, ignore_index=ignore_index,
                            class_weights=class_weights)
    else:
        raise ValueError(f"Unknown loss mode: {mode}")


# ─── 4. Dataset: Stanford Background ─────────────────────────────────────────

class StanfordBackgroundDataset(Dataset):
    """
    Stanford Background Dataset
    https://dags.stanford.edu/projects/scenedataset.html

    Structure:
        iheS_images/    *.jpg
        iheS_labels/    *.regions.txt   (space-separated integer label grid)
    8 classes: sky, tree, road, grass, water, building, mountain, foreground

    Fallback: any folder with  images/  and  labels/  sub-dirs.
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    NUM_CLASSES = 8
    CLASS_NAMES = [
        "sky", "tree", "road", "grass",
        "water", "building", "mountain", "foreground",
    ]
    IMG_SIZE = 256

    def __init__(self, root: str, split: str = "train",
                 val_ratio: float = 0.15, seed: int = 42):
        self.root = Path(root)
        self.size = self.IMG_SIZE
        self.split = split

        # Locate images and masks
        img_dir  = self._find_dir(["iheS_images", "images", "imgs"])
        mask_dir = self._find_dir([
            "iheS_labels",
            "labels_raw",
            "labels",
            "masks",
            "annotations",
        ])

        if img_dir is None or mask_dir is None:
            nested_root = self._find_nested_root()
            if nested_root is not None:
                self.root = nested_root
                img_dir  = self._find_dir(["iheS_images", "images", "imgs"])
                mask_dir = self._find_dir([
                    "iheS_labels",
                    "labels_raw",
                    "labels",
                    "masks",
                    "annotations",
                ])

        if img_dir is None or mask_dir is None:
            children = ", ".join(sorted(p.name for p in self.root.iterdir() if p.is_dir()))
            raise FileNotFoundError(
                f"Could not find image/label directories under {root}.\n"
                f"Found sub-folders: {children or '<none>'}\n"
                f"Expected image folder like 'images' and label folder like 'labels_raw'."
            )

        image_exts = (".jpg", ".jpeg", ".png")
        all_imgs = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in image_exts
        )
        all_masks = []
        for img in all_imgs:
            mask = None
            for suffix in (".regions.txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                candidate = mask_dir / (img.stem + suffix)
                if candidate.exists():
                    mask = candidate
                    break
            if mask is not None:
                all_masks.append(mask)
            else:
                all_masks.append(None)

        # Filter pairs where mask exists
        pairs = [(i, m) for i, m in zip(all_imgs, all_masks) if m is not None]

        # Deterministic train/val split
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(pairs))
        n_val = max(1, int(len(pairs) * val_ratio))
        if split == "val":
            pairs = [pairs[i] for i in idx[:n_val]]
        else:
            pairs = [pairs[i] for i in idx[n_val:]]

        self.pairs = pairs
        self.img_tf  = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )
        print(f"[Dataset] Stanford Background  split={split}  n={len(self.pairs)}")

    def _find_dir(self, candidates):
        if not self.root.is_dir():
            return None
        existing = {p.name.lower(): p for p in self.root.iterdir() if p.is_dir()}
        for name in candidates:
            d = existing.get(name.lower())
            if d is not None:
                return d
        return None

    def _find_nested_root(self):
        if not self.root.is_dir():
            return None
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            names = {p.name.lower() for p in child.iterdir() if p.is_dir()}
            has_images = bool(names & {"ihes_images", "images", "imgs"})
            has_labels = bool(names & {
                "ihes_labels",
                "labels_raw",
                "labels",
                "masks",
                "annotations",
            })
            if has_images and has_labels:
                return child
        return None

    def _load_mask_txt(self, path: Path) -> np.ndarray:
        """Load .regions.txt (rows of space-separated ints)."""
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append([int(v) for v in line.split()])
        return np.array(rows, dtype=np.int64)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")

        if mask_path.suffix == ".txt":
            mask_np = self._load_mask_txt(mask_path)
        else:
            mask_np = np.array(Image.open(mask_path))

        mask_img = Image.fromarray(mask_np.astype(np.int32), mode="I")

        if self.split == "train":
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.7:
                img = self.color_jitter(img)

        img_t = self.img_tf(img)
        mask_img = mask_img.resize((self.size, self.size),
                                   resample=Image.NEAREST)
        mask_t = torch.from_numpy(np.array(mask_img)).long()

        # Stanford marks unknown pixels with negative labels; keep them ignored.
        valid_mask = (mask_t >= 0) & (mask_t < self.NUM_CLASSES)
        mask_t = torch.where(valid_mask, mask_t, torch.full_like(mask_t, 255))

        return img_t, mask_t


def get_dataloaders(root: str, batch_size: int = 8):
    train_ds = StanfordBackgroundDataset(root, split="train")
    val_ds   = StanfordBackgroundDataset(root, split="val")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
    return train_dl, val_dl


# ─── 5. mIoU Metric ───────────────────────────────────────────────────────────

def compute_miou(preds: torch.Tensor, targets: torch.Tensor,
                 num_classes: int, ignore_index: int = 255) -> float:
    """
    preds:   [B, H, W]  integer class predictions
    targets: [B, H, W]  integer ground truth
    Returns: mean IoU over valid classes.
    """
    ious = []
    preds   = preds.view(-1)
    targets = targets.view(-1)
    mask    = targets != ignore_index
    preds   = preds[mask]
    targets = targets[mask]

    for cls in range(num_classes):
        pred_c   = preds   == cls
        target_c = targets == cls
        inter    = (pred_c & target_c).sum().item()
        union    = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(inter / union)

    return float(np.mean(ious)) if ious else 0.0


def compute_intersection_union(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
):
    preds = preds.view(-1)
    targets = targets.view(-1)
    mask = targets != ignore_index
    preds = preds[mask]
    targets = targets[mask]

    intersections = torch.zeros(num_classes, dtype=torch.float64)
    unions = torch.zeros(num_classes, dtype=torch.float64)
    for cls in range(num_classes):
        pred_c = preds == cls
        target_c = targets == cls
        intersections[cls] = (pred_c & target_c).sum().item()
        unions[cls] = (pred_c | target_c).sum().item()
    return intersections, unions


def iou_from_intersection_union(intersections, unions):
    per_class = []
    for inter, union in zip(intersections, unions):
        per_class.append(float(inter / union) if union > 0 else float("nan"))
    valid = [v for v in per_class if not np.isnan(v)]
    return per_class, float(np.mean(valid)) if valid else 0.0


# ─── 6. Training Loop ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device,
                    epoch: int, num_classes: int, run=None):
    model.train()
    total_loss, total_miou, n = 0.0, 0.0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total_miou += compute_miou(preds, masks, num_classes)
        total_loss += loss.item()
        n += 1

    avg_loss = total_loss / n
    avg_miou = total_miou / n
    if run:
        run.log({"train/loss": avg_loss, "train/mIoU": avg_miou, "epoch": epoch})
    return avg_loss, avg_miou


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch: int,
             num_classes: int, run=None, split: str = "val",
             return_per_class: bool = False):
    model.eval()
    total_loss, total_miou, n = 0.0, 0.0, 0
    total_inter = torch.zeros(num_classes, dtype=torch.float64)
    total_union = torch.zeros(num_classes, dtype=torch.float64)
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss   = criterion(logits, masks)
        preds  = logits.argmax(dim=1)
        total_miou += compute_miou(preds, masks, num_classes)
        inter, union = compute_intersection_union(
            preds.cpu(), masks.cpu(), num_classes
        )
        total_inter += inter
        total_union += union
        total_loss += loss.item()
        n += 1

    avg_loss = total_loss / n
    per_class_iou, dataset_miou = iou_from_intersection_union(total_inter, total_union)
    avg_miou = dataset_miou
    if run:
        run.log({f"{split}/loss": avg_loss, f"{split}/mIoU": avg_miou,
                 "epoch": epoch})
    if return_per_class:
        return avg_loss, avg_miou, per_class_iou
    return avg_loss, avg_miou


# ─── 7. Experiment Runner ─────────────────────────────────────────────────────

def estimate_class_weights(dataset: StanfordBackgroundDataset,
                           num_classes: int) -> torch.Tensor:
    """Median-frequency style weights from the training masks."""
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, mask_path in dataset.pairs:
        if mask_path.suffix == ".txt":
            mask_np = dataset._load_mask_txt(mask_path)
        else:
            mask_np = np.array(Image.open(mask_path))
        mask = torch.from_numpy(mask_np).long()
        valid = (mask >= 0) & (mask < num_classes)
        if valid.any():
            counts += torch.bincount(mask[valid].view(-1),
                                     minlength=num_classes).double()
    freq = counts / counts.sum().clamp_min(1.0)
    nonzero = freq[freq > 0]
    median = nonzero.median() if len(nonzero) else torch.tensor(1.0)
    weights = median / freq.clamp_min(1e-6)
    return (weights / weights.mean()).float()


def run_experiment(cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Experiment: {cfg['loss_mode'].upper()}  |  Device: {device}")
    print(f"{'='*60}")

    run = None
    if SWANLAB_AVAILABLE and cfg.get("use_tracking", False):
        run = swanlab.init(
            project="unet-segmentation",
            experiment_name=f"unet_{cfg['loss_mode']}",
            config=cfg,
        )

    train_dl, val_dl = get_dataloaders(cfg["data_dir"], cfg["batch_size"])
    NUM_CLASSES = StanfordBackgroundDataset.NUM_CLASSES

    model     = UNet(in_channels=3, num_classes=NUM_CLASSES,
                     base_ch=cfg.get("base_ch", 32),
                     use_attention=cfg.get("attention_unet", False)).to(device)
    class_weights = None
    if cfg.get("class_weighted", False) and cfg["loss_mode"] in {"ce", "combined"}:
        class_weights = estimate_class_weights(train_dl.dataset, NUM_CLASSES).to(device)
        print(f"[Loss] class weights: {[round(v, 4) for v in class_weights.detach().cpu().tolist()]}")
    criterion = get_loss_fn(
        cfg["loss_mode"],
        NUM_CLASSES,
        combined_alpha=cfg.get("combined_alpha", 0.5),
        class_weights=class_weights,
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=1e-4)
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=cfg["epochs"], power=0.9)

    best_val_miou = 0.0
    best_state = None
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_miou = train_one_epoch(
            model, train_dl, criterion, optimizer, device, epoch, NUM_CLASSES, run
        )
        va_loss, va_miou = evaluate(
            model, val_dl, criterion, device, epoch, NUM_CLASSES, run
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{cfg['epochs']}  "
              f"train_loss={tr_loss:.4f}  train_mIoU={tr_miou:.4f}  "
              f"val_loss={va_loss:.4f}  val_mIoU={va_miou:.4f}  "
              f"({elapsed:.1f}s)")
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_mIoU": tr_miou,
            "val_loss": va_loss,   "val_mIoU": va_miou,
        })
        if va_miou > best_val_miou:
            best_val_miou = va_miou
            best_state = copy.deepcopy(model.state_dict())

    os.makedirs(cfg["output_dir"], exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    _, final_val_miou, per_class_iou = evaluate(
        model, val_dl, criterion, device, cfg["epochs"],
        NUM_CLASSES, run=None, split="val", return_per_class=True,
    )
    per_class_path = os.path.join(
        cfg["output_dir"], f"per_class_iou_{cfg['loss_mode']}.csv"
    )
    import csv
    with open(per_class_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "iou"])
        writer.writeheader()
        for cls_id, iou in enumerate(per_class_iou):
            writer.writerow({
                "class_id": cls_id,
                "class_name": StanfordBackgroundDataset.CLASS_NAMES[cls_id],
                "iou": "" if np.isnan(iou) else f"{iou:.4f}",
            })

    ckpt_path = os.path.join(
        cfg["output_dir"], f"unet_{cfg['loss_mode']}_best.pth"
    )
    torch.save({
        "config": cfg,
        "state_dict": best_state,
        "best_val_mIoU": final_val_miou,
        "per_class_iou": {
            StanfordBackgroundDataset.CLASS_NAMES[i]: None if np.isnan(iou) else float(iou)
            for i, iou in enumerate(per_class_iou)
        },
    }, ckpt_path)

    result_path = os.path.join(
        cfg["output_dir"], f"result_{cfg['loss_mode']}.json"
    )
    with open(result_path, "w") as f:
        json.dump({
            "config": cfg,
            "best_val_mIoU": final_val_miou,
            "checkpoint": ckpt_path,
            "per_class_iou": {
                StanfordBackgroundDataset.CLASS_NAMES[i]: None if np.isnan(iou) else float(iou)
                for i, iou in enumerate(per_class_iou)
            },
        }, f, indent=2)

    hist_path = os.path.join(cfg["output_dir"],
                             f"history_{cfg['loss_mode']}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    if run:
        run.finish()

    print(f"[Result] loss_mode={cfg['loss_mode']}  best_val_mIoU={final_val_miou:.4f}")
    print(f"[Result] best checkpoint saved to {ckpt_path}")
    print(f"[Result] per-class IoU saved to {per_class_path}")
    return {"loss_mode": cfg["loss_mode"], "best_val_mIoU": final_val_miou}


def run_comparison(data_dir: str, output_dir: str,
                   epochs: int = 40, batch_size: int = 8, lr: float = 1e-3,
                   combined_alpha: float = 0.5,
                   class_weighted: bool = False,
                   attention_unet: bool = False):
    """
    Run all three loss configurations and produce a comparison summary.
    """
    import csv
    base_cfg = dict(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        base_ch=32,
        combined_alpha=combined_alpha,
        class_weighted=class_weighted,
        attention_unet=attention_unet,
        use_tracking=SWANLAB_AVAILABLE,
    )

    results = []
    for loss_mode in ["ce", "dice", "combined"]:
        cfg = {**base_cfg, "loss_mode": loss_mode}
        result = run_experiment(cfg)
        results.append(result)

    # Summary
    print(f"\n{'─'*50}")
    print(f"  Loss Function Comparison — Stanford Background")
    print(f"{'─'*50}")
    for r in results:
        print(f"  {r['loss_mode']:10s}  best_val_mIoU = {r['best_val_mIoU']:.4f}")

    summary_path = os.path.join(output_dir, "loss_comparison.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["loss_mode", "best_val_mIoU"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[Summary] saved to {summary_path}")
    return results


# ─── 8. CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3: U-Net Segmentation")
    parser.add_argument("--data_dir",   required=True,
                        help="Path to Stanford Background Dataset root")
    parser.add_argument("--output_dir", default="./outputs/task3")
    parser.add_argument("--mode",
                        choices=["ce", "dice", "combined", "all"],
                        default="all",
                        help="Loss mode. 'all' runs all three and compares.")
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--base_ch",    type=int,   default=32,
                        help="Base channel width of U-Net (default 32)")
    parser.add_argument("--combined_alpha", type=float, default=0.5,
                        help="CE weight in combined loss: alpha*CE + (1-alpha)*Dice")
    parser.add_argument("--class_weighted", action="store_true",
                        help="Use median-frequency class weights for CE terms")
    parser.add_argument("--attention_unet", action="store_true",
                        help="Use attention gates on U-Net skip connections")
    args = parser.parse_args()

    if args.mode == "all":
        run_comparison(args.data_dir, args.output_dir,
                       args.epochs, args.batch_size, args.lr,
                       args.combined_alpha,
                       args.class_weighted,
                       args.attention_unet)
    else:
        cfg = dict(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            loss_mode=args.mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            base_ch=args.base_ch,
            combined_alpha=args.combined_alpha,
            class_weighted=args.class_weighted,
            attention_unet=args.attention_unet,
            use_tracking=SWANLAB_AVAILABLE,
        )
        run_experiment(cfg)
