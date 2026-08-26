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


# from pybaseutils import font_utils
# font_utils.set_pyplot_font(font="SimHei")

import wandb
from ultralytics import settings, YOLO

settings.update({"wandb": True})
wandb.login(key="fdb78e84a884b4d2f51a52025b9f59c806d5e2f3")
# wandb.init(project="coco")


weight = r"D:\python_work\ultralytics\weights\yolov8s.pt"
cfg = r"D:\python_work\ultralytics\models\yolov8s.yaml"
# datasets = "/determined/alluxio/public/dengxiongshi/datasets/person_car/20250211/person_car.yaml"
datasets = r"D:\python_work\yolov5\datasets\coco128\coco.yaml"
epoch = 20
imgsz = 640
batch_size = 4
device = [0]
# project = ROOT / "runs/train/detect"
project = "test"
name = "yolov8s_chinese"
optimizer = "SGD"
workers = 0

# Add W&B callback for Ultralytics
# add_wandb_callback(model, enable_model_checkpointing=True)


model = YOLO(model=cfg, task="detect", verbose=True)
model.load(weights=weight)


results = model.train(data=datasets, epochs=epoch, patience=100, imgsz=imgsz, batch=batch_size, device=device,
                      project=project, name=name, optimizer=optimizer, lr0=0.01, workers=workers, resume=False)

'''RESUME'''
# weights = "/data/ultralytics/runs/train/detect/person_car/yolov8s_20250211/weights/last.pt"
# model = YOLO(model=weights)
# results = model.train(resume=True)