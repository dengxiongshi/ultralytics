import sys
import platform
import os
from pathlib import Path

import cv2
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import DetectionModel

from ultralytics import YOLO

weight = "/mnt/d/python_work/ultralytics/weights/yolov8s-seg.pt"
source = "/mnt/d/python_work/ultralytics/ultralytics/assets/zidane.jpg"
imgsz = 640
device = 0
project = ROOT / "runs/test"
name = "result"
# save_dir = r"C:\Users\dengxs\Desktop\test1\test\result"
conf = 0.45
classes = [0, 2]

model = YOLO(model=weight, task="segment")

results = model.predict(source=source, imgsz=imgsz, device=device, save=False,
                        project=project, name=name, conf=conf)

for img_path, result in zip([source], results):
    print(f"Shape of result.masks.data: {None if result.masks is None else result.masks.data.shape}")
    # print(result.masks)

    # YOLO's own render (masks, boxes, etc.)
    rendered = result.plot()
    H0, W0 = rendered.shape[:2]
    print(f"H0: {H0}, W0: {W0}")
    print(f"Processing image: {img_path} (original size: {W0}x{H0})")

    masks = result.masks
    num_instances = 0

    if masks is not None and masks.data is not None:
        mask_stack = masks.data.cpu().numpy()  # (N, Hm, Wm)
        num_instances = mask_stack.shape[0]

        resized_masks = []
        for m in mask_stack:
            m = (m > 0.5).astype(np.uint8)
            if m.shape != (H0, W0):
                m = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(m)
        resized_masks = np.stack(resized_masks, axis=0) if resized_masks else None
    else:
        resized_masks = None

    if resized_masks is not None:
        union_mask = np.any(resized_masks, axis=0)
        img_data = union_mask * 255
        save_dir = result.save_dir
        os.makedirs(save_dir, exist_ok=True)
        base = Path(img_path).stem
        out_img_path = f"{base}_pred.jpg"
        cv2.imwrite(os.path.join(save_dir, out_img_path), img_data)
        print(f"Saved rendered image (masks only) to: {out_img_path}")
