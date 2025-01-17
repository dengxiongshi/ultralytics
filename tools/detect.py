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



weight = "/data/ultralytics/runs/train/detect/coco/yolo11s/weights/best.onnx"
source = "/data/yolov5/data/images/bus.jpg"
device = 0
project = "/data/ultralytics/runs/detect/detect/coco"
name = "yolo11s"

model = YOLO(model=weight)

results = model.predict(source=source, imgsz=[384, 640], device=device, save=True, project=project, name=name)

