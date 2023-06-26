import argparse 
import torch

INPUT_SHAPE = (640, 640)

def main():
    args = build_argument_parser()

    onnx_model_path = args.onnx_path
    opset = args.opset_version
    input_names = args.model_input_name
    output_names = args.model_output_name

    torch_model = torch.hub.load("ultralytics/yolov5:v7.0", "yolov5s")
    torch_model = torch_model.eval()

    print("Start ONNX export...")
    torch.onnx.export(
        torch_model,
        torch.zeros(1, 3, *INPUT_SHAPE),
        onnx_model_path, 
        input_names = [input_names],
        output_names = [output_names],
        opset_version = opset,
    )

    print(f"Completed ONNX export >> {onnx_model_path}")


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path',type=str, default = "./Result.onnx")
    parser.add_argument('--opset_version', type=int, default=13, help='the ONNX version to export the model to') 
    parser.add_argument('--model_input_name', type=str, default='images', help="the model's input name") 
    parser.add_argument('--model_output_name', type=str, default='output', help="the model's output name")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
