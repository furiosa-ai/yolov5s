import argparse
import os
import random
from tqdm import tqdm

import onnx
from onnx.utils import Extractor

from furiosa.optimizer import optimize_model
from furiosa.quantizer import quantize, Calibrator, CalibrationMethod
from utils.preprocess import load_input

INPUT_SHAPE = (640, 640)


def main():
    args = build_argument_parser()

    onnx_path = args.onnx_path
    dfg_path = args.dfg_path
    opset = args.opset_version
    calib_data = args.calib_data
    calib_count = args.calib_count
    input_name = args.model_input_name
    model_repo = args.model_repo

    assert model_repo in [
        "ultralytics",
        "openmmlab",
    ], "Unsupported torchmodel for yolov5s structure!!"

    f32_onnx_model = onnx.load_model(onnx_path)
    extracted_onnx_model = extract_model(f32_onnx_model, input_name, model_repo)
    optimized_onnx_model = optimize_model(
        model=extracted_onnx_model,
        opset_version=opset,
        input_shapes={input_name: [1, 3, *INPUT_SHAPE]},
    )

    calib_data_names = os.listdir(calib_data)
    calib_data_names = random.choices(calib_data_names, k=calib_count)

    calibrator = Calibrator(optimized_onnx_model, CalibrationMethod.SQNR_ASYM)

    for image_name in tqdm(calib_data_names, desc="Calibrating", unit="image", mininterval=0.5):
        image_path = os.path.join(calib_data, image_name)
        image, _ = load_input(image_path, new_shape=INPUT_SHAPE)
        calibrator.collect_data([[image]])

    ranges = calibrator.compute_range()

    i8_onnx_model = quantize(optimized_onnx_model, ranges)

    with open(dfg_path, "wb") as f:
        f.write(bytes(i8_onnx_model))

    print(f"Completed quantinization >> {dfg_path}")


def extract_model(model: onnx.ModelProto, input_name, model_repo):
    """Cut off the post-processing components."""
    input_to_shape = [(input_name, (1, 3, *INPUT_SHAPE))]

    # TODO: The cut points below were chosen so that the postprocess() function can follow the
    # structure of the original PyTorch code as closely as possible. We need to experiment more with
    # different cut points to achieve the best combination of speed and accuracy.

    if model_repo == "ultralytics":
        output_to_shape = (
            # between /model/model/model.24/m.0/Conv and /model/model/model.24/Reshape
            ("/model/model/model.24/m.0/Conv_output_0", (1, 255, 80, 80)),
            # between /model/model/model.24/m.1/Conv and /model/model/model.24/Reshape_2
            ("/model/model/model.24/m.1/Conv_output_0", (1, 255, 40, 40)),
            # between /model/model/model.24/m.2/Conv and /model/model/model.24/Reshape_4
            ("/model/model/model.24/m.2/Conv_output_0", (1, 255, 20, 20)),
        )
    else:
        output_to_shape = (
            ("/head_module/convs_pred.0/Conv_output_0", (1, 255, 80, 80)),
            ("/head_module/convs_pred.1/Conv_output_0", (1, 255, 40, 40)),
            ("/head_module/convs_pred.2/Conv_output_0", (1, 255, 20, 20)),
        )
    input_to_shape = {
        tensor_name: [
            onnx.TensorShapeProto.Dimension(dim_value=dimension_size) for dimension_size in shape
        ]
        for tensor_name, shape in input_to_shape
    }
    output_to_shape = {
        tensor_name: [
            onnx.TensorShapeProto.Dimension(dim_value=dimension_size) for dimension_size in shape
        ]
        for tensor_name, shape in output_to_shape
    }

    extracted_model = Extractor(model).extract_model(
        input_names=list(input_to_shape), output_names=list(output_to_shape)
    )

    for value_info in extracted_model.graph.input:
        del value_info.type.tensor_type.shape.dim[:]
        value_info.type.tensor_type.shape.dim.extend(input_to_shape[value_info.name])
    for value_info in extracted_model.graph.output:
        del value_info.type.tensor_type.shape.dim[:]
        value_info.type.tensor_type.shape.dim.extend(output_to_shape[value_info.name])

    return extracted_model


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, default="Result.onnx", help="Path to onnx file")
    parser.add_argument(
        "--model_repo",
        type=str,
        default="ultralytics",
        help="repository = [ultralytics, openmmlab]",
    )
    parser.add_argument("--dfg_path", type=str, default="./Result.dfg", help="Path to i8 onnx file")
    parser.add_argument(
        "--opset_version",
        type=int,
        default=13,
        help="the ONNX version to export the model to",
    )
    parser.add_argument(
        "--calib_data",
        type=str,
        default="./coco/train2017",
        help="Path to calibration data containing image files",
    )
    parser.add_argument(
        "--calib_count",
        default=10,
        type=int,
        help="How many images to use for calibration",
    )
    parser.add_argument(
        "--model_input_name", type=str, default="images", help="the model's input name"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
