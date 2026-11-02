import sys
import platform
import os
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from ultralytics import RTDETR
# import wandb
# from ultralytics import settings
#
# wandb.login(key="fdb78e84a884b4d2f51a52025b9f59c806d5e2f3")
# settings.update({"wandb": True})

cfg = r"G:\python_work\ultralytics\models\RT-DETR\rtdetr-r18.yaml"
weight = r"G:\python_work\ultralytics\weights\rtdetr-l.pt"
datasets = r"G:\python_work\yolov5\datasets\coco128\coco.yaml"
epoch = 20
imgsz = 640
batch_size = 4
device = [0]
project = "rt-detr"
name = "rtdetr-r18"
optimizer = "SGD"
workers = 0

model = RTDETR(cfg)
# model.load(weights=weight)
model.info()

results = model.train(data=datasets, epochs=epoch, patience=100, imgsz=imgsz, batch=batch_size, device=device,
                      project=project, name=name, optimizer=optimizer, lr0=0.01, workers=workers, resume=False)