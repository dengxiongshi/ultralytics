import argparse
import math
import pathlib
import sys
import platform
import os
import warnings
from pathlib import Path

import onnx
import thop
import torch
from prettytable import PrettyTable

from thop import profile


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.models import YOLO
from ultralytics.nn import DetectionModel, SegmentationModel
from ultralytics.utils.torch_utils import profile_ops


def make_divisible(x, divisor):
    """Adjusts `x` to be divisible by `divisor`, returning the nearest greater or equal value."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def check_img_size(imgsz, s=32, floor=0):
    """Adjusts image size to be divisible by stride `s`, supports int or list/tuple input, returns adjusted size."""
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(f"WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}")
    return new_size


def check(model, f):
    """
    Args:
        model: pt model
        f: onnx file

    Returns:

    """
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

def export(model, inputs, save_file, simplify=True):
    output_names = ["output0", "output1"] if isinstance(model, SegmentationModel) else ["output0"]
    torch.onnx.export(
        model=model,
        args=inputs,
        f=save_file,
        input_names=["input"],  # 输入张量名称（后续 TensorRT 绑定用）
        output_names=output_names,  # 输出张量名称
        opset_version=11,  # OPset 版本（推荐 11-13，兼容 TensorRT 7+）
        do_constant_folding=True,  # 折叠常量，优化模型体积和速度
        verbose=True  # 不打印详细日志（True 用于调试）
    )

    # Checks
    onnx_model = onnx.load(save_file)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    onnx.save(onnx_model, save_file)

    if simplify:
        import onnxsim
        print(f"simplifying with onnx-simplifier {onnxsim.__version__}...")
        try:
            model_simp, check = onnxsim.simplify(save_file)
            assert check, 'assert check failed'
            print((onnxsim.model_info.print_simplifying_info(onnx_model, model_simp)))
            onnx.save(model_simp, save_file)
        except Exception as e:
            assert e, 'simplifier failure'


def parse_opt():
    parser = argparse.ArgumentParser(description='SeAFusion Net eval process')
    parser.add_argument('--model_path', default='models/yolov8/segment/yolov8-seg.yaml', help='fusion weight checkpoint',
                        type=pathlib.Path)  # weight/default.pth
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--channels", type=int, default=1, help="channels")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[512, 640], help="image (h, w)")
    parser.add_argument("--opset", type=int, default=11, help="ONNX: opset version")
    parser.add_argument('--simplify', default=True, action='store_true', help='ONNX: simplify model')
    opt = parser.parse_args()

    print(vars(opt))

    return opt


def main(args):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    imgsz = [check_img_size(x) for x in args.imgsz]  # verify img_size are gs-multiples
    batch_size, channel = args.batch_size, args.channels
    input = torch.randn(batch_size, channel,  *imgsz).to(device)
    detect = SegmentationModel(cfg=args.model_path, ch=channel, nc=2)
    detect.eval().to(device)

    export_onnx_file_path = str(args.model_path).replace(args.model_path.suffix, '.onnx')
    export(detect, input, export_onnx_file_path, args.simplify)


if __name__ == '__main__':
    warnings.filterwarnings(
        "once",
        message="Constant folding not applied",
    )
    args = parse_opt()
    main(args)

# result = profile_ops(input=input, ops=[detect], n=3, device=device)

# model = YOLO(cfg)  # select your model.pt path
# model = model.model
# model.fuse()
# total_flops, total_params, layers = profile(model, [input], verbose=True, ret_layer_info=True)
# FLOPs, Params = thop.clever_format([total_flops * 2 / batch_size, total_params], "%.3f")
# table = PrettyTable()
# table.title = f'Model Flops:{FLOPs} Params:{Params}'
# table.field_names = ['Layer ID', "FLOPs", "Params"]
# for layer_id in layers['model'][2]:
#     data = layers['model'][2][layer_id]
#     FLOPs, Params = thop.clever_format([data[0] * 2 / batch_size, data[1]], "%.3f")
#     table.add_row([layer_id, FLOPs, Params])
# print(table)

# model_file = r"/home/dxs/share/docker_share/dxs/yolov5/data_seaship/best.onnx"
#
# model = AutoBackend(weights=model_file)


