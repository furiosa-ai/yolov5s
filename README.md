# Yolov5s with Furiosa-SDK
This repository gives an example of optimally compiling and running a YOLOv5s model using the options provided by the [Furiosa SDK](https://furiosa-ai.github.io/docs/latest/ko/index.html).


## Setup
### Setup Environment
```sh
git clone git@github.com:furiosa-ai/yolov5s.git
cd yolov5s
conda activate <env>
pip install -r requirements.txt
```

### Calibration and Evaluation Dataset
Download COCO dataset from [here](https://cocodataset.org/#download) or prepare own dataset for calibration

## Export ONNX
Convert torch model to onnx model. We used pretrained torch model provided [ultralytics YoloV5s](https://github.com/ultralytics/yolov5).
```sh
  --onnx_path ONNX_PATH
  --opset_version OPSET_VERSION
                        the ONNX version to export the model to
  --model_input_name MODEL_INPUT_NAME
                        the model's input name
  --model_output_name MODEL_OUTPUT_NAME
                        the model's output name
```

## Furiosa Quantization
Convert f32 onnx model to i8 onnx model using ```furiosa.quantinizer```. This involves a process for cutting off the post-processing elements.

```sh
python furiosa_quantize.py -h
  --onnx_path ONNX_PATH
                        Path to onnx file
  --dfg_path DFG_PATH   Path to i8 onnx file
  --opset_version OPSET_VERSION
                        the ONNX version to export the model to
  --calib_data CALIB_DATA
                        Path to calibration data containing image files
  --calib_count CALIB_COUNT
                        How many images to use for calibration
  --model_input_name MODEL_INPUT_NAME
                        the model's input name
```

## Run
Create a session using the quantized model obtained from ```furiosa_quantize.py```. Use the sessions you create to make inferences on your test data set.

```sh
python furiosa_eval.py -h
  --dfg_path DFG_PATH   Path to dfg file
  --eval_data_path EVAL_DATA_PATH
                        Path to evaluation data containing image files
  --eval_count EVAL_COUNT
                        How many images to use for evaluation
  --output_path OUTPUT_PATH
                        Path to result image
  ...
```
