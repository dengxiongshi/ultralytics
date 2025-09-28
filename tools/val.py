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
from ultralytics.utils.torch_utils import profile

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import DetectionModel

from ultralytics import YOLO

weight = "/data/ultralytics/runs/train/detect/person_car/yolov8s/weights/best.onnx"
datasets = "/determined/alluxio/public/dengxiongshi/datasets/person_car/20250211/person_car.yaml"
batch_size = 32
device = 0
split = "test"  # Determines the dataset split to use for validation (val, test, or train)
project = "runs/val/person_car"
name = "yolov8s"

model = YOLO(model=weight)

results = model.val(data=datasets, imgsz=640, batch=batch_size, conf=0.001, iou=0.6, device=device, split=split,
                    project=project, name=name)

# Evaluate model performance on the validation set
# metrics = model.val()
