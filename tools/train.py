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


weight = "/data/ultralytics/weights/yolov8s.pt"
cfg = "/data/ultralytics/models/yolov8s.yaml"
datasets = "/determined/alluxio/public/dengxiongshi/datasets/person_car/20250211/person_car.yaml"
epoch = 2000
batch_size = 144
device = [0,1,2,3]
project = "runs/train/detect/person_car"
name = "yolov8s_20250211"
optimizer = "SGD"

# Add W&B callback for Ultralytics
# add_wandb_callback(model, enable_model_checkpointing=True)


model = YOLO(model=cfg, task="detect", verbose=True)
model.load(weights=weight)


results = model.train(data=datasets, epochs=epoch, patience=100, imgsz=640, batch=batch_size, device=device,
                      project=project, name=name, optimizer=optimizer, lr0=0.01, resume=False)

'''RESUME'''
# weights = "/data/ultralytics/runs/train/detect/person_car/yolov8s_20250211/weights/last.pt"
# model = YOLO(model=weights)
# results = model.train(resume=True)