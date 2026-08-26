import sys
import platform
import os
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import DetectionModel

from ultralytics import YOLO

weight = r"D:\python_work\ultralytics\runs\detect\test\yolov8s_chinese\weights\best.pt"
source = r"D:\python_work\ultralytics\ultralytics\assets\zidane.jpg"
imgsz = 640
device = 0
project = ROOT / "runs/test"
name = "result"
# save_dir = r"C:\Users\dengxs\Desktop\test1\test\result"
conf = 0.25
iou = 0.7
batch = 4  # 指定推理的批次大小（仅当源为 目录、视频文件或 .txt 文件 时有效）
stream = True  # 通过返回 Results 对象的生成器而不是一次将所有帧加载到内存中，实现针对长视频或大量图像的高内存效率处理
# classes = [0, 2]

model = YOLO(model=weight, task="detect")

results = model.predict(source=source, imgsz=imgsz, device=device, save=True, show=False,
                        project=project, name=name, conf=conf, batch=batch, stream=stream)

# 关键：遍历生成器，触发实际推理和自动保存
for res in results:
    # 循环内可以写你的业务逻辑（比如读取检测框），不写也能正常保存结果
    # 每处理完一帧/一张图，自动释放对应内存
    # 可选：每帧强制回收碎片内存，解决你之前的坏帧崩溃问题
    torch.cuda.empty_cache()

# 全部跑完后释放模型
del model
torch.cuda.empty_cache()
print("全部推理完成")