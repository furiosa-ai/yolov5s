# Yolov5s with Furiosa-SDK
This repository gives an example of optimally compiling and running a YOLOv5s model using the options provided by the [Furiosa SDK](https://furiosa-ai.github.io/docs/latest/ko/index.html).

## Setup
### Setup Environment
```sh
git clone git@github.com:furiosa-ai/yolov5s.git
cd yolov5s
conda create -n demo python=3.9
conda activate demo
pip install -r requirements.txt
```

if you want to use yolov5s provided by [openmmlab](https://github.com/open-mmlab/mmyolo/tree/main), please install additional packages as [installation] (https://github.com/open-mmlab/mmyolo/tree/main#%EF%B8%8F-installation-). Then, download mmyolo and weights file as bellow.
```sh
mim install mmyolo
wget -P ./datas https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth
```

### Calibration and Evaluation Dataset
Download COCO dataset from [here](https://cocodataset.org/#download) or prepare own dataset for calibration
#### Example
```sh
wget http://images.cocodataset.org/zips/val2017.zip 
unzip val2017.zip -d ./coco/
```

## Export ONNX
Convert torch model to onnx model. 
### Example for ultralytics
```sh
python onnx_export.py --onnx_path=./yolov5s.onnx --model_repo=ultralytics --opset_version=13 --model_input_name=images --model_output_name=outputs
```

### Example for openmmlab
```sh
python onnx_export.py --onnx_path=./yolov5s.onnx --model_repo=openmmlab --opset_version=13 --model_input_name=images --model_output_name=outputs
```

## Furiosa Quantization
Convert f32 onnx model to i8 onnx model using ```furiosa.quantinizer```. This involves a process for cutting off the post-processing elements.

### Example for ultralytics
```sh
python furiosa_quantize.py --onnx_path=./yolov5s.onnx --model_repo=ultralytics --dfg_path=./yolov5s.dfg --opset_version=13 --calib_data=./coco/val2017 --calib_count=10 --model_input_name=images
```

### Example for openmmlab
```sh
python furiosa_quantize.py --onnx_path=./yolov5s.onnx --model_repo=openmmlab --dfg_path=./yolov5s.dfg --opset_version=13 --calib_data=./coco/val2017 --calib_count=10 --model_input_name=images
```

```sh
# Argument
python furiosa_quantize.py -h
  --onnx_path ONNX_PATH
                        Path to onnx file
  --model_repo MODEL_REPO
                        repository = [ultralytics, openmmlab]
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

### Example
```sh
python furiosa_eval.py --dfg_path=./yolov5s.dfg --eval_data_path=./coco/val2017 --eval_count=10 --output_path=./output
```

```sh
# Argument
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
