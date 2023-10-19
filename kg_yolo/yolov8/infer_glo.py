import os
import cv2
import json
import glob
import time
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import pycocotools.mask as mask_util
from PIL import Image
import matplotlib.pyplot as plt
from ensemble_boxes import *
from ultralytics import YOLO
import itertools
import shutil

model_paths = ['/home/yiw/kg_yolo/yolov8/runs/segment/train0_f0_0.396/weights/best_f0.pt',
               '/home/yiw/kg_yolo/yolov8/runs/segment/train1_f1_0.402/weights/best_f1.pt',
               '/home/yiw/kg_yolo/yolov8/runs/segment/train4_f3_0.397/weights/best_f3.pt',
               '/home/yiw/kg_yolo/yolov8/runs/segment/train2_f2_0.380/weights/best_f2.pt',
               '/home/yiw/kg_yolo/yolov8/best_f4.pt',]

models = []
for model_path in model_paths:
    model = YOLO(model_path)
    models.append(model)

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def nms_predictions(classes, scores, bboxes, polygons, 
                    iou_th=.6, shape=(1024, 1024)):
    he, wd = shape[0], shape[1]
    boxes_list = [[[x[0] / wd, x[1] / he, x[2] / wd, x[3] / he] for x in bboxes]]
    scores_list = [[x for x in scores]]
    classes_list = [[x for x in classes]]
    nms_bboxes, nms_scores, nms_classes = non_maximum_weighted(
        boxes_list, 
        scores_list, 
        classes_list, 
        weights=None,
        iou_thr=iou_th,
        skip_box_thr=0.0001,
    )
    nms_polygons = []
    for s in nms_scores:
        nms_polygons.append(polygons[scores.index(s)])
    nms_scores, nms_classes, nms_polygons = zip(*sorted(zip(nms_scores, nms_classes, nms_polygons), reverse=True))
    return nms_classes, nms_scores, nms_polygons

test_names = glob.glob("/home/yiw/kg_yolo/glo/*png")
print('test images:', len(test_names))

os.makedirs(f'yolov8/pseduo_glo', exist_ok=True)

idx = 0
coco_annotations = []
for test_name in tqdm(test_names):
    test_name = os.path.basename(test_name)
    img = Image.open(f"/home/yiw/kg_yolo/glo/{test_name}")
    img = np.array(img)
    h, w, _ = img.shape
    classes, scores, bboxes, polygons = [], [], [], []

    for model in models:
        test_result = model(img, conf=0.55)
        if len(test_result) > 0 and len(test_result[0].boxes.conf) > 0:
            classes.extend([0] * len(test_result[0].boxes.conf))
            scores.extend(test_result[0].boxes.conf.cpu().numpy().tolist())
            bboxes.extend(test_result[0].boxes.data.cpu().numpy()[:, :4].tolist())
            polygons.extend(test_result[0].masks.xy)

    if len(scores) > 0:
        classes, scores, polygons = nms_predictions(classes, scores, bboxes, polygons)

    if len(polygons) > 0:
        shutil.copyfile(f"/home/yiw/kg_yolo/glo/{test_name}", f"yolov8/pseduo_glo/{test_name}")
        with open(f"yolov8/pseduo_glo/{test_name.split('.')[0]}.txt", 'w') as text_file:
            for polygon in polygons:
                flat_mask_polygon = list(itertools.chain(*polygon))
                array = np.array(flat_mask_polygon)/1024
                text_file.write(f'0 {" ".join(map(str, array))}\n')
