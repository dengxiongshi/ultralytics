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

weight = r"D:\python_work\ultralytics\weights\yolov10s.pt"
imgsz = [384, 640]
# cfg = "/data/ultralytics/weights/yolo11s.yaml"
# datasets = "/data/yolov5/datasets/coco128/coco.yaml"
# epoch = 600
# batch_size = 32
device = 0
# project = "runs/train/detect/coco"
# name = "yolo11s"
# optimizer = "SGD"
opset = 16

model = YOLO(model=weight)

model.export(format="onnx", imgsz=imgsz, device=device, simplify=True, opset=opset)
