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

weight = ROOT / "weights/yolo11x.pt"
source = r"C:\Users\dengxs\Desktop\test1\test\1.mp4"
imgsz = 640
device = 0
project = r"C:\Users\dengxs\Desktop\test1\test"
name = "result"
# save_dir = r"C:\Users\dengxs\Desktop\test1\test\result"
conf = 0.45
classes = [0, 2]

model = YOLO(model=weight)

results = model.predict(source=source, imgsz=imgsz, device=device, save=True,
                        project=project, name=name, conf=conf, classes=classes)
