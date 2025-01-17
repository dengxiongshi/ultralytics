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



weight = "/data/ultralytics/runs/train/detect/coco/yolov8s/weights/best.pt"
# cfg = "/data/ultralytics/weights/yolo11s.yaml"
# datasets = "/data/yolov5/datasets/coco128/coco.yaml"
# epoch = 600
# batch_size = 32
# device = 0
# project = "runs/train/detect/coco"
# name = "yolo11s"
# optimizer = "SGD"


model = YOLO(model=weight)

model.export(format="onnx", imgsz=[384, 640], simplify=True, opset=11)

