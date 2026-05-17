# HW2 Deep Learning and Spatial Intelligence

This repository contains code for three computer vision tasks:

1. Flower classification on Oxford 102 Flowers
2. Vehicle detection, multi-object tracking and cross-line counting
3. U-Net semantic segmentation on Stanford Background Dataset

The recommended entry point for reviewing saved results is:

```bash
python replay_all_results.py --outputs ../outputs
```

## Saved Weights

The trained weights are not included in this GitHub repository because of file size.

Download the `train` weights folder from Google Drive:

[Download trained weights](https://drive.google.com/file/d/13URjE0vUfw2KumBYQMmOdy2Qu7gePvby/view?usp=drive_link)

After downloading, place it under:

```text
outputs/train/
```

Expected structure:

```text
outputs/
  train/
    task1/
      *_best.pth
    task2/
      weights/
        best.pt
        last.pt
    task3/
      unet_ce_best.pth
      unet_dice_best.pth
      unet_combined_best.pth
```

## 1. Replay and Show All Results

`replay_all_results.py` reads saved CSV/JSON results and optionally reloads
weights to recompute metrics.

Lightweight summary:

```bash
python replay_all_results.py --outputs ../outputs
```

Recompute Task 1 test accuracy:

```bash
python replay_all_results.py \
  --outputs ../outputs \
  --weights_dir ../outputs/train \
  --flowers_dir ../data/flowers102 \
  --recompute_task1
```

Recompute Task 3 validation mIoU:

```bash
python replay_all_results.py \
  --outputs ../outputs \
  --weights_dir ../outputs/train \
  --stanford_dir ../data/stanfordBackground \
  --recompute_task3
```

Replay Task 2 tracking:

```bash
python replay_all_results.py \
  --outputs ../outputs \
  --weights_dir ../outputs/train \
  --video_path ../data/test_video.mp4 \
  --rerun_task2
```

Regenerate visualization figures:

```bash
python replay_all_results.py \
  --outputs ../outputs \
  --visualize \
  --video_frame
```

Full replay:

```bash
python replay_all_results.py \
  --outputs ../outputs \
  --weights_dir ../outputs/train \
  --flowers_dir ../data/flowers102 \
  --stanford_dir ../data/stanfordBackground \
  --video_path ../data/test_video.mp4 \
  --recompute_task1 \
  --recompute_task3 \
  --rerun_task2 \
  --visualize \
  --video_frame
```

## 2. Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

GPU is recommended for all training scripts.

Optional SwanLab logging can be disabled with:

```bash
--disable_tracking
```

or:

```bash
export SWANLAB_MODE=disabled
```

## 3. Task 1: Flower Classification

Script:

```text
train.py
```

Task:

- Dataset: Oxford 102 Category Flowers
- Model families: ResNet-18, ResNet-34, EfficientNet-B0, ConvNeXt-Tiny, Swin-T
- Attention modules: SE-block and CBAM
- Metric: accuracy

Run the original ablation suite:

```bash
python main.py \
  --tasks task1 \
  --flowers_dir ../data/flowers102 \
  --output_dir ../outputs \
  --task1_mode ablation \
  --disable_tracking
```

Run one backbone:

```bash
python main.py \
  --tasks task1 \
  --flowers_dir ../data/flowers102 \
  --output_dir ../outputs \
  --task1_mode single \
  --task1_arch convnext_tiny \
  --task1_exp_name convnext_tiny_pretrained \
  --disable_tracking
```

Supported `--task1_arch` values:

```text
resnet18
resnet34
efficientnet_b0
convnext_tiny
swin_t
```

Evaluate all Task 1 checkpoints on the test split:

```bash
python evaluate_task1_checkpoints.py \
  --data_dir ../data/flowers102 \
  --output_dir ../outputs/task1
```

Output:

```text
outputs/task1/task1_full_summary.csv
outputs/task1/task1_full_summary.json
```

### CBAM Tuning

Script:

```text
tune_cbam_task1.py
```

Run preset CBAM tuning experiments:

```bash
python tune_cbam_task1.py \
  --data_dir ../data/flowers102 \
  --output_dir ../outputs/task1_cbam_tuning \
  --epochs 30 \
  --batch_size 64
```

Run one custom CBAM configuration:

```bash
python tune_cbam_task1.py \
  --data_dir ../data/flowers102 \
  --output_dir ../outputs/task1_cbam_tuning \
  --single \
  --exp_name cbam_l34_custom \
  --cbam_layers layer3,layer4 \
  --cbam_reduction 16 \
  --cbam_kernel 7 \
  --cbam_lr 5e-4 \
  --mixup_alpha 0 \
  --cutmix_alpha 0
```

Output:

```text
outputs/task1_cbam_tuning/cbam_tuning_summary.csv
```

## 4. Task 2: Vehicle Detection, Tracking and Counting

Script:

```text
detect_track.py
```

Task:

- Dataset: Road Vehicle Images Dataset in YOLO format
- Detector: YOLOv8
- Tracker: ByteTrack
- Metrics: precision, recall, mAP50, mAP50-95, cross-line count

Expected dataset structure:

```text
data/road_vehicles/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

Fine-tune YOLOv8:

```bash
python main.py \
  --tasks task2 \
  --vehicles_dir ../data/road_vehicles \
  --output_dir ../outputs \
  --task2_phase finetune \
  --task2_model_size yolov8n
```

Run tracking and cross-line counting:

```bash
python main.py \
  --tasks task2 \
  --output_dir ../outputs \
  --task2_phase track \
  --task2_model_path ../outputs/train/task2/weights/best.pt \
  --video_path ../data/test_video.mp4
```

Fine-tune and track in one command:

```bash
python main.py \
  --tasks task2 \
  --vehicles_dir ../data/road_vehicles \
  --output_dir ../outputs \
  --task2_phase all \
  --task2_model_size yolov8n \
  --video_path ../data/test_video.mp4
```

Optional manual counting line:

```bash
--line_x1 0 --line_y1 1536 --line_x2 2160 --line_y2 1536
```

Output:

```text
outputs/task2/vehicle_finetune/results.csv
outputs/task2/vehicle_finetune/weights/best.pt
outputs/task2/tracked_output.mp4
outputs/task2/tracking_summary.json
outputs/task2/occlusion_frames/
```

## 5. Task 3: U-Net Semantic Segmentation

Script:

```text
unet_train.py
```

Task:

- Dataset: Stanford Background Dataset
- Model: U-Net implemented from scratch
- Losses: Cross-Entropy, Dice, CE + Dice
- Metric: mIoU and per-class IoU

Expected dataset structure:

```text
data/stanfordBackground/
  images/
  labels_raw/
    *.regions.txt
  labels_colored/
  labels_class_dict.csv
  metadata.csv
```

Run all three loss experiments:

```bash
python main.py \
  --tasks task3 \
  --stanford_dir ../data/stanfordBackground \
  --output_dir ../outputs \
  --task3_mode all
```

Run a single loss:

```bash
python main.py \
  --tasks task3 \
  --stanford_dir ../data/stanfordBackground \
  --output_dir ../outputs \
  --task3_mode combined
```

Run the improved Attention U-Net + class-weighted setting:

```bash
python main.py \
  --tasks task3 \
  --stanford_dir ../data/stanfordBackground \
  --output_dir ../outputs \
  --task3_mode combined \
  --task3_attention_unet \
  --task3_class_weighted
```

Output:

```text
outputs/task3/history_ce.json
outputs/task3/history_dice.json
outputs/task3/history_combined.json
outputs/task3/loss_comparison.csv
outputs/task3/per_class_iou_ce.csv
outputs/task3/per_class_iou_dice.csv
outputs/task3/per_class_iou_combined.csv
outputs/task3/unet_ce_best.pth
outputs/task3/unet_dice_best.pth
outputs/task3/unet_combined_best.pth
```

## 6. Run Everything End-to-End

Script:

```text
run_all.py
```

Run all tasks:

```bash
python run_all.py \
  --flowers_dir ../data/flowers102 \
  --vehicles_dir ../data/road_vehicles \
  --stanford_dir ../data/stanfordBackground \
  --video_path ../data/test_video.mp4 \
  --output_dir ../outputs
```

Dry run:

```bash
python run_all.py \
  --flowers_dir ../data/flowers102 \
  --vehicles_dir ../data/road_vehicles \
  --stanford_dir ../data/stanfordBackground \
  --video_path ../data/test_video.mp4 \
  --output_dir ../outputs \
  --dry_run
```

Skip selected parts:

```bash
python run_all.py --video_path ../data/test_video.mp4 --skip_task1
python run_all.py --video_path ../data/test_video.mp4 --skip_improved_task3
```

## 7. Visualization

Script:

```text
visualize_results.py
```

Generate report and slide figures:

```bash
python visualize_results.py \
  --outputs ../outputs \
  --video_frame
```

Output:

```text
outputs/visualizations/
  task1_test_accuracy.png
  task1_val_accuracy_curves.png
  task2_yolo_map_curve.png
  task2_yolo_loss_curve.png
  task2_tracking_summary.png
  task2_tracked_midframe.jpg
  task2_occlusion_montage.jpg
  task3_loss_comparison.png
  task3_val_miou_curves.png
  task3_per_class_iou_all_losses.png
  summary.md
```

## 8. Report Generation

Generate Word report from saved outputs:

```bash
python generate_updated_report.py
```

Output:

```text
../HW2_report_updated.docx
```

## 9. Notes

- Use `/` paths on Linux, for example `../data/test_video.mp4`.
- `test_video.mp4` is an input video for Task 2. The model-generated output is `outputs/task2/tracked_output.mp4`.
- If GitHub does not include weights, download `outputs/train` from the Google Drive link above before using `replay_all_results.py`.
- For fast inspection, start with `python replay_all_results.py --outputs ../outputs`.

