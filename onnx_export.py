import argparse
import torch

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.runner import Runner
from mmyolo.registry import RUNNERS

INPUT_SHAPE = (640, 640)


def load_openmmlab_model(config, weights):
    cfg = Config.fromfile(config)
    cfg.load_from = weights
    cfg.test_pipeline[1]["scale"] = INPUT_SHAPE

    runner = Runner.from_cfg(cfg)
    runner.call_hook("before_run")
    runner.load_or_resume()
    return runner.model


def main():
    args = build_argument_parser()

    onnx_model_path = args.onnx_path
    opset = args.opset_version
    input_names = args.model_input_name
    output_names = args.model_output_name
    model_repo = args.model_repo

    assert model_repo in [
        "ultralytics",
        "openmmlab",
    ], "Unsupported torchmodel for yolov5s structure!!"

    if model_repo == "ultralytics":
        torch_model = torch.hub.load("ultralytics/yolov5:v7.0", "yolov5s")
    elif model_repo == "openmmlab":
        config = args.config
        weights = args.weights
        torch_model = load_openmmlab_model(config, weights)

    torch_model = torch_model.eval()

    print("Start ONNX export...")
    torch.onnx.export(
        torch_model,
        torch.zeros(1, 3, *INPUT_SHAPE),
        onnx_model_path,
        input_names=[input_names],
        output_names=[output_names],
        opset_version=opset,
    )
    print(f"Completed ONNX export >> {onnx_model_path}")


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, default="./Result.onnx")
    parser.add_argument(
        "--model_repo",
        type=str,
        default="ultralytics",
        help="repository = [ultralytics, openmmlab]",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./datas/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py",
        help="Path to config file to convert openmmlab torch model",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="./datas/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth",
        help="Path to weight file to convert openmmlab torch model",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=13,
        help="the ONNX version to export the model to",
    )
    parser.add_argument(
        "--model_input_name", type=str, default="images", help="the model's input name"
    )
    parser.add_argument(
        "--model_output_name",
        type=str,
        default="output",
        help="the model's output name",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
