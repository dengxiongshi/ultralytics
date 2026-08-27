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


import wandb
from ultralytics import settings

wandb.login(key="fdb78e84a884b4d2f51a52025b9f59c806d5e2f3")
# wandb.init(project="coco")
settings.update({"wandb": True})

from ultralytics import YOLO


weight = r"G:\python_work\ultralytics\weights\yolov8n-cls.pt"
cfg = r"G:\python_work\ultralytics\models\yolov8\classify\yolov8n-cls.yaml"
# datasets = "/determined/alluxio/public/dengxiongshi/datasets/person_car/20250211/person_car.yaml"
datasets = r"G:\python_work\ultralytics\datasets\traindata\datasets.yaml"
epoch = 100
imgsz = 224
batch_size = 1
device = [0]
project = ROOT / "train/classify"
name = "test"
workers = 0
optimizer = "SGD"
multi_label = True  # Set to True for multi-label classification, False for single-label classification

# Add W&B callback for Ultralytics
# add_wandb_callback(model, enable_model_checkpointing=True)


model = YOLO(model=cfg, task="classify", verbose=True)
model.load(weights=weight)


results = model.train(data=datasets, epochs=epoch, patience=100, imgsz=imgsz, batch=batch_size, device=device, multi_label=multi_label,
                      project=project, name=name, optimizer=optimizer, lr0=0.01, workers=0, resume=False)

'''RESUME'''
# weights = "/data/ultralytics/runs/train/detect/person_car/yolov8s_20250211/weights/last.pt"
# model = YOLO(model=weights)
# results = model.train(resume=True)