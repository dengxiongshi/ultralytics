import os
import sys
from pathlib import Path

import amct_onnx as amct
import numpy as np
from amct_onnx.common.auto_calibration import AutoCalibrationEvaluatorBase

from tools.onnx_val import DetectionValidator

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def fitness(x):
    """Calculates fitness of a model using weighted sum of metrics P, R, mAP@0.5, mAP@0.5:0.95."""
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def onnx_forward(onnx_file, data, batch_size=1, imgs=[640, 640]):
    # data = "/home/dxs/share/docker_share/dxs/yolov8/quantity/person_car_quantization_images/person_car.yaml"
    project = ROOT / "runs/quantization"
    name = "test"

    device = "cpu"
    task = "detect"
    mode = "val"
    split = "test"
    workers = 1
    plots = False

    args = dict(model=onnx_file, data=data, imgsz=imgs, device=device, batch=batch_size, task=task,
                mode=mode, split=split, workers=workers, plots=plots)
    validator = DetectionValidator(args=args)
    validator()

    return validator


# You need to implement the AutoCalibrationEvaluator's calibration(), evaluate() and metric_eval() funcs
class AutoCalibrationEvaluator(AutoCalibrationEvaluatorBase):
    """ subclass of AutoCalibrationEvaluatorBase"""

    def __init__(self, target_loss, data, batch_num, imgsz):
        super(AutoCalibrationEvaluator, self).__init__()
        self.target_loss = target_loss
        self.data = data
        self.batch_num = batch_num
        self.imgsz = imgsz

    def calibration(self, model_file):
        """ implement the calibration function of AutoCalibrationEvaluatorBase
            calibration() need to finish the calibration inference procedure
            so the inference batch num need to >= the batch_num pass to create_quant_config
        """
        onnx_forward(model_file, self.data, batch_size=1, imgs=self.imgsz)

    def evaluate(self, model_file):
        """ implement the evaluate function of AutoCalibrationEvaluatorBase
            params: model_file in .onnx
            return: the accuracy of input model on the eval dataset, or other metric which
                    can describe the 'accuracy' of model
        """
        validator = onnx_forward(model_file, self.data, batch_size=1, imgs=self.imgsz)
        results = [validator.metrics.box.mp, validator.metrics.box.mr, validator.metrics.box.map50, validator.metrics.box.map]
        results = np.array(results).reshape(1, -1)
        fi = fitness(results)
        return fi

    def metric_eval(self, original_metric, new_metric):
        """ implement the metric_eval function of AutoCalibrationEvaluatorBase
            params: original_metric: the returned accuracy of evaluate() on non quantized model
                    new_metric: the returned accuracy of evaluate() on fake quant model
            return:
                   [0]: whether the accuracy loss between non quantized model and fake quant model
                        can satisfy the requirement
                   [1]: the accuracy loss between non quantized model and fake quant model
        """
        loss = original_metric - new_metric
        if loss * 100 < self.target_loss:
            return True, loss
        return False, loss


def main():
    data = "/home/dxs/share/docker_share/dxs/yolov8/quantity/person_car_quantization_images/person_car.yaml"
    model_file = "/home/dxs/share/docker_share/dxs/yolov8/quantity/A15_20250121.onnx"
    imgsz = [384, 640]
    batch_num = 4

    PATH = os.path.realpath('./')
    TMP = "/home/dxs/share/docker_share/dxs/yolov8/quantity/accuracy_based"
    os.makedirs(TMP, exist_ok=True)

    print('[INFO] Do original model test:')
    ori_validator = onnx_forward(model_file, data, batch_size=batch_num, imgs=imgsz)

    config_json_file = os.path.join(TMP, 'config.json')
    skip_layers = []

    amct.create_quant_config(
        config_file=config_json_file, model_file=model_file, skip_layers=skip_layers, batch_num=batch_num,
        activation_offset=True, config_defination=None)

    # 1. step1 create quant config json file
    scale_offset_record_file = os.path.join(TMP, 'scale_offset_record.txt')
    result_path = os.path.join(TMP, 'results/yolov8')

    # 2. step2 construct the instance of AutoCalibrationEvaluator
    evaluator = AutoCalibrationEvaluator(target_loss=0.4, data=data, batch_num=batch_num, imgsz=imgsz)

    # 3. step3 using the accuracy_based_auto_calibration to quantized the model
    amct.accuracy_based_auto_calibration(
        model_file=model_file,
        model_evaluator=evaluator,
        config_file=config_json_file,
        record_file=scale_offset_record_file,
        save_dir=result_path,
        strategy='BinarySearch',
        sensitivity='CosineSimilarity'
    )

    # 4. step4 run fake_quant model test
    print('[INFO] Do quantized model test:')
    quant_validator = onnx_forward('%s_%s' % (result_path, 'fake_quant_model.onnx'), data, 1, imgsz)
    print('[INFO] yolov8s before quantize mAP50:{:>10} mAP50-95:{:>10}'.format(ori_validator.metrics.box.map50, ori_validator.metrics.box.map))
    print('[INFO] yolov8s after quantize  mAP50:{:>10} mAP50-95:{:>10}'.format(quant_validator.metrics.box.map50, quant_validator.metrics.box.map))


if __name__ == '__main__':
    main()