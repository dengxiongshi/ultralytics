import torch

from ultralytics.nn import DetectionModel

from ultralytics.utils.torch_utils import profile


cfg = r"D:\python_work\ultralytics\weights\yolo11s.yaml"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
im = torch.rand(1, 3, 640, 640).to(device)
detect = DetectionModel(cfg=cfg)
result = profile(input=im, ops=[detect], n=3)