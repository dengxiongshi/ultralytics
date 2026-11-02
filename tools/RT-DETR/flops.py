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

from ultralytics import RTDETR
from ultralytics.models import YOLO
from ultralytics.models.yolo import detect
from ultralytics.nn import DetectionModel, SegmentationModel
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import profile_ops


cfg = r"G:\python_work\ultralytics\models\RT-DETR\rtdetr-r18.yaml"
batch_size = 1
input = torch.randn((1, 3, 640, 640))
device = "0"

# result = profile_ops(input=input, ops=[detect], n=3, device=device)

# model = RTDETR(cfg)  # select your model.pt path
model = RTDETRDetectionModel(cfg, ch=3, nc=80)
model.eval()
# Display model information (optional)
model.info(detailed=False)
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


