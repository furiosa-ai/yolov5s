import argparse
import os
import numpy as np
import random
from tqdm import tqdm

import cv2
from utils.preprocess import load_input
from utils.postprocess import load_output, non_max_suppression, draw_bbox

import furiosa.runtime.session

INPUT_SHAPE = (640, 640)


def main():
    args = build_argument_parser()
    dfg_path = args.dfg_path
    conf_thres = args.conf_thres
    iou_thres = args.iou_thres
    eval_data_path = args.eval_data_path
    eval_count = args.eval_count
    output_path = args.output_path

    image_names = os.listdir(eval_data_path)
    image_names = random.choices(image_names, k=eval_count)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    with furiosa.runtime.session.create(dfg_path) as session:
        for image_name in tqdm(image_names, desc="Evaluating", unit="image", mininterval=0.5):
            image_path = os.path.join(eval_data_path, image_name)
            img, preproc_param = load_input(image_path, new_shape=INPUT_SHAPE)

            outputs = session.run([img]).numpy()
            outputs = load_output(outputs)
            outputs = non_max_suppression(
                outputs,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                agnostic=False,
                max_det=300,
            )

            assert len(outputs) == 1, f"{len(outputs)=}"

            predictions = outputs[0]

            num_predictions = predictions.shape[0]
            if num_predictions == 0:
                continue

            bboxed_img = draw_bbox(image_path, predictions, preproc_param)
            cv2.imwrite(os.path.join(output_path, image_name), bboxed_img)

    print(f"Completed Evaluation")


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfg_path", type=str, default="Result.dfg", help="Path to dfg file")
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default="./coco/val2017",
        help="Path to evaluation data containing image files",
    )
    parser.add_argument(
        "--eval_count",
        default=10,
        type=int,
        help="How many images to use for evaluation",
    )
    parser.add_argument("--conf_thres", type=float, default=0.65)
    parser.add_argument("--iou_thres", type=float, default=0.35)
    parser.add_argument("--output_path", type=str, default="./output", help="Path to result image")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
