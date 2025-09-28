import sys
import platform
import os
from pathlib import Path

from tools.onnx_infer import AutoBackend

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch

from ultralytics.nn import DetectionModel

from ultralytics.utils.torch_utils import profile

cfg = r"models/yolov8s.yaml"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
im = torch.rand(1, 3, 640, 640).to(device)
detect = DetectionModel(cfg=cfg)
result = profile(input=im, ops=[detect], n=3)

model_file = r"/home/dxs/share/docker_share/dxs/yolov5/data_seaship/best.onnx"

model = AutoBackend(weights=model_file)
