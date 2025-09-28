import argparse

import amct_onnx as amct

import os
import sys
from pathlib import Path

from tools.onnx_val import DetectionValidator

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def onnx_forward(onnx_file, data, batch_size=1, imgs=[640, 640]):
    # data = "/home/dxs/share/docker_share/dxs/yolov8/quantity/person_car_quantization_images/person_car.yaml"
    project = ROOT / "runs/quantization"
    name = "test"

    device = "cpu"
    task = "detect"
    mode = "val"
    split = "test"
    workers = 0
    plots = False

    args = dict(model=onnx_file, data=data, imgsz=imgs, device=device, batch=batch_size, task=task,
                mode=mode, split=split, workers=workers, plots=plots)
    validator = DetectionValidator(args=args)
    validator()

    return validator


PARSER = argparse.ArgumentParser(description='amct_onnx yolov8 quantization sample.')
PARSER.add_argument('--nuq', default=False, help='whether use nuq')
ARGS = PARSER.parse_args()

if ARGS.nuq:
    TMP = "/home/dxs/share/docker_share/dxs/yolov8/quantity/nuq"
else:
    TMP = "/home/dxs/share/docker_share/dxs/yolov8/quantity/uq"
os.makedirs(TMP, exist_ok=True)

data = "/home/dxs/share/docker_share/dxs/yolov8/quantity/person_car_quantization_images/person_car.yaml"
model_file = "/home/dxs/share/docker_share/dxs/yolov8/quantity/A15_20250121.onnx"
imgs=[384, 640]

config_file = os.path.join(TMP, "config.json")
skip_layers = []
batch_num = 16
config_defination = os.path.join(TMP, "nuq_quant.cfg")

if ARGS.nuq:
    amct.create_quant_config(config_file=config_file, model_file=model_file, skip_layers=skip_layers,
                             batch_num=batch_num, activation_offset=True, config_defination=config_defination)
else:
    amct.create_quant_config(config_file=config_file, model_file=model_file, skip_layers=skip_layers,
                             batch_num=batch_num, activation_offset=True, config_defination=None)

# Phase1: do conv+bn fusion, weights calibration and generate
#         calibration model
scale_offset_record_file = os.path.join(TMP, "record.txt")
modified_model = os.path.join(TMP, "modified_model.onnx")
amct.quantize_model(config_file=config_file, model_file=model_file, modified_onnx_file=modified_model, record_file=scale_offset_record_file)
onnx_forward(modified_model, data=data, imgs=imgs)

# Phase3: save final model, one for onnx do fake quant test, one
#         deploy model for ATC
result_path = os.path.join(TMP, 'yolov8')
amct.save_model(modified_model, scale_offset_record_file, result_path)
print('[INFO] Do quantized model test:')
validator = onnx_forward('%s_%s' % (result_path, 'fake_quant_model.onnx'), data=data, imgs=imgs)

