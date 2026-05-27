from pathlib import Path
import sys

import numpy as np
import os
import cv2
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics import YOLO


def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else x


def load_yolo_mask(txt_path, img_shape):
    H, W = img_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    if not os.path.exists(txt_path):
        return mask
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cls = int(parts[0])
            coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
            coords[:, 0] *= W
            coords[:, 1] *= H
            coords = coords.astype(np.int32)
            cv2.fillPoly(mask, [coords], 1)
    return mask


def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0


def validate(model_path, data_dir, debug=False, out_dir="debug_outputs", mask_threshold=0.5, lane_class_id=0):
    model = YOLO(model_path, task="segment")

    image_dir = os.path.join(data_dir, "images", "val")
    label_dir = os.path.join(data_dir, "labels", "val")
    os.makedirs(out_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(image_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    ious = []
    processed = 0
    skipped = 0

    for img_file in tqdm(img_files, desc="Validating"):
        img_path = os.path.join(image_dir, img_file)
        txt_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue
        H, W = img.shape[:2]

        gt_mask = load_yolo_mask(txt_path, (H, W))

        try:
            results = model.predict(img_path, retina_masks=True, verbose=False)
            pred_mask = np.zeros((H, W), dtype=np.uint8)

            for r in results:
                if r.masks is not None and r.masks.data is not None:
                    md = to_numpy(r.masks.data)
                    if hasattr(r, 'boxes') and r.boxes is not None and hasattr(r.boxes, 'cls'):
                        cls_ids = to_numpy(r.boxes.cls)
                    else:
                        cls_ids = np.full(len(md), lane_class_id)
                    for i, m in enumerate(md):
                        if i < len(cls_ids):
                            cls_id = int(cls_ids[i])
                            if cls_id != lane_class_id:
                                continue
                        if m.shape != (H, W):
                            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                        m_bin = (m > mask_threshold).astype(np.uint8)
                        pred_mask = np.maximum(pred_mask, m_bin)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            skipped += 1
            continue

        iou = compute_iou(pred_mask, gt_mask)
        ious.append(iou)
        processed += 1

        if debug:
            overlay = img.copy()
            overlay[gt_mask == 1] = [0, 255, 0]
            overlay[pred_mask == 1] = [0, 0, 255]
            overlap = (gt_mask == 1) & (pred_mask == 1)
            overlay[overlap] = [0, 255, 255]
            gt_pixels = np.sum(gt_mask == 1)
            pred_pixels = np.sum(pred_mask == 1)
            cv2.putText(overlay, f"IoU: {iou:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(overlay, f"GT: {gt_pixels}, Pred: {pred_pixels}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            out_path = os.path.join(out_dir, img_file)
            cv2.imwrite(out_path, overlay)

    if not ious:
        print("No valid predictions found!")
        return 0.0

    mean_iou = np.mean(ious)
    print(f"\nValidation Results:")
    print(f"Processed: {processed} images")
    print(f"Skipped: {skipped} images")
    print(f"Mean IoU (Lane): {mean_iou:.4f}")
    print(f"Min IoU: {np.min(ious):.4f}")
    print(f"Max IoU: {np.max(ious):.4f}")
    print(f"Std IoU: {np.std(ious):.4f}")

    return mean_iou
