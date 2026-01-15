import sys
from pathlib import Path

import cv2
import onnxruntime
import numpy as np
import torch
from numpy import ndarray
from typing import List, Tuple



FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH


def softmax(x: ndarray, axis: int = -1) -> ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    y = e_x / e_x.sum(axis=axis, keepdims=True)
    return y


def sigmoid(x: ndarray) -> ndarray:
    return 1. / (1. + np.exp(-x))


def postprocess(feats: List[ndarray],
                conf_thres: float = 0.25,
                reg_max: int = 16) -> Tuple[List, List, List]:
    dfl = np.arange(0, reg_max, dtype=np.float32)
    feats = [feat[0] for feat in feats]
    scores_proposals = []
    boxes_proposals = []
    labels_proposals = []

    for i in range(len(feats)):
        stride = 8 << i
        score_feat = feats[i][..., :80]
        bbox_feat = feats[i][..., 80:]

        hIdx, cIdx = np.where(score_feat > conf_thres)
        num_proposals = hIdx.size
        if not num_proposals:
            continue

        scores = score_feat[hIdx, cIdx]
        boxes = bbox_feat[hIdx].reshape(-1, 4, reg_max)
        boxes = softmax(boxes, -1) @ dfl

        for k in range(num_proposals):
            h = hIdx[k]
            score = scores[k]
            label = cIdx[k]
            x1, y1, x2, y2 = boxes[k]

            x1 = (w + 0.5 - x1) * stride
            y1 = (h + 0.5 - y1) * stride
            x2 = (w + 0.5 + x2) * stride
            y2 = (h + 0.5 + y2) * stride

            scores_proposals.append(score)
            boxes_proposals.append(np.array([x1, y1, x2, y2], dtype=np.float32))
            labels_proposals.append(label)

    return boxes_proposals, scores_proposals, labels_proposals


img_path = r"D:\python_work\ultralytics\ultralytics\assets\bus.jpg"

onnx_path = r"E:\soft\modified_onnx\modified_yolov10s.onnx"

img = cv2.imread(img_path)
img = cv2.resize(img, (640, 384))

data = np.ascontiguousarray(
    img[:, :, ::-1].transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']
session = onnxruntime.InferenceSession(onnx_path, providers=providers)

outputs = session.run(None, {'images': data})

boxes, scores, labels = postprocess(outputs)

for box, score, label in zip(boxes, scores, labels):
    box = np.clip(box, 0, 640).round().astype(np.int32)
    cv2.rectangle(img, box[:2], box[2:], (0, 255, 0), 2)

cv2.imshow('image', img)
cv2.waitKey(0)