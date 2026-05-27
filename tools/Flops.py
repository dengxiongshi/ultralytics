import argparse
import math
import pathlib
import sys
import platform
import os
import warnings
from pathlib import Path

import onnx
import thop
import torch
from prettytable import PrettyTable

from thop import profile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.models import YOLO
from ultralytics.models.yolo import detect
from ultralytics.nn import DetectionModel, SegmentationModel
from ultralytics.utils.torch_utils import profile_ops


cfg = "/mnt/d/python_work/ultralytics/models/yolov8/segment/yolov8_shufflenetv2.yaml"
batch_size = 1
input = torch.randn((1, 3, 640, 640))
device = "0"

# result = profile_ops(input=input, ops=[detect], n=3, device=device)

model = YOLO(cfg, task="segment", verbose=True)  # select your model.pt path
model = model.model
model.fuse()
total_flops, total_params, layers = profile(model, [input], verbose=True, ret_layer_info=True)
FLOPs, Params = thop.clever_format([total_flops * 2 / batch_size, total_params], "%.3f")
table = PrettyTable()
table.title = f'Model Flops:{FLOPs} Params:{Params}'
table.field_names = ['Layer ID', "FLOPs", "Params"]
for layer_id in layers['model'][2]:
    data = layers['model'][2][layer_id]
    FLOPs, Params = thop.clever_format([data[0] * 2 / batch_size, data[1]], "%.3f")
    table.add_row([layer_id, FLOPs, Params])
print(table)

# model_file = r"/home/dxs/share/docker_share/dxs/yolov5/data_seaship/best.onnx"
#
# model = AutoBackend(weights=model_file)


