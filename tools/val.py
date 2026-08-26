import sys
import platform
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import DetectionModel

from ultralytics import YOLO

weight = "/home/dxs/snap/ultralytics/runs/detect/fish/yolov8n/weights/best.pt"
datasets = "/home/dxs/snap/train_data/dataset.yaml"
batch_size = 64
device = "0"
split = "test"  # Determines the dataset split to use for validation (val, test, or train)
project = "fish"
name = "yolov8n"
workers = 0

model = YOLO(model=weight)

results = model.val(data=datasets, imgsz=640, batch=batch_size, conf=0.001, iou=0.6, device=device, split=split,
                    project=project, name=name, workers=workers)

# Evaluate model performance on the validation set
# metrics = model.val()

# Assuming results contain both predicted masks and ground truth
# for result in results:
#     # Get predicted masks and ground truth masks
#     pred_masks = result.masks.pred  # Predicted masks from the model
#     gt_masks = result.masks.gt  # Ground truth masks from the dataset
#
#     # Calculate IoU for each class
#     ious = []
#     for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
#         intersection = (pred_mask & gt_mask).sum()  # Logical AND
#         union = (pred_mask | gt_mask).sum()  # Logical OR
#         iou = intersection / union if union > 0 else 0
#         ious.append(iou)
#         print(f"Class {i}: IoU = {iou:.4f}")
#
#     # Calculate mean IoU across all classes
#     mean_iou = sum(ious) / len(ious)
#     print(f"Mean IoU: {mean_iou:.4f}")