import argparse
import math
import pathlib
import sys
import platform
import os
import warnings
from pathlib import Path

import onnx
import torch

from ultralytics.utils import LOGGER

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from ultralytics import RTDETR

weight = r"G:\python_work\ultralytics\runs\detect\rt-detr\rtdetr-r18-2\weights\best.pt"
imgsz = [384, 640]
# cfg = "/mnt/d/python_work/ultralytics/models/yolo26/segment/yolo26s_xsmall.yaml"
# datasets = "/data/yolov5/datasets/coco128/coco.yaml"
# epoch = 600
# batch_size = 32
device = "0"
# project = "runs/train/detect/coco"
# name = "yolo11s"
# optimizer = "SGD"
opset = 16
task = "detect"

model = RTDETR(weight)

model.export(format="onnx", imgsz=imgsz, device=device, simplify=True, opset=opset)