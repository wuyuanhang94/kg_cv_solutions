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
from ensemble_boxes import *
import json, itertools
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
                    iou_th=.2, shape=(512, 512)):
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
idx = 0
coco_annotations = []

image_id = []
file_names = []

for test_name in tqdm(test_names):
    test_name = os.path.basename(test_name)
    img = Image.open(f"/home/yiw/kg_yolo/glo/{test_name}")
    img = np.array(img)
    h, w, _ = img.shape
    classes, scores, bboxes, masks = [], [], [], []

    for model in models:
        test_result = model(img, conf=0.55)
        if len(test_result) > 0 and len(test_result[0].boxes.conf) > 0:
            classes.extend([0] * len(test_result[0].boxes.conf))
            scores.extend(test_result[0].boxes.conf.cpu().numpy().tolist())
            bboxes.extend(test_result[0].boxes.data.cpu().numpy()[:, :4].tolist())
            mks = []
            for i, p in enumerate(test_result[0].masks.data):
                mask = cv2.resize(p.cpu().numpy(), (512, 512))
                mask = mask.astype(np.bool_)
                mks.append(mask)
            masks.extend(mks)

    if len(scores) > 0:
        classes, scores, masks = nms_predictions(classes, scores, bboxes, masks)

    if len(masks) > 0:
        file_names.append(f"/home/yiw/kg_yolo/glo/{test_name}")
        image_id.append(test_name)
        for mk in masks:
            ys, xs = np.where(mk)
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            enc = binary_mask_to_rle(mk)
            seg = {
                'segmentation':enc, 
                'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)],
                'bbox_mode': 1,
                'area': int(np.sum(mk)),
                'image_id': test_name.split(".")[0], 
                'category_id': 1,
                'iscrowd': 0, 
                'id': idx
            }
            idx += 1
            coco_annotations.append(seg)

cats =[{'name': "b", 'id':1}]
images = [{'id':os.path.basename(test_name).split('.')[0], 'width': 512, 'height': 512, 'file_name': test_name} for test_name in test_names]
final_coco_annos = {'categories':cats, 'images':images,'annotations': coco_annotations}

with open(f'pseduo_glo_coco.json', 'w', encoding='utf-8') as f:
    json.dump(final_coco_annos, f, ensure_ascii=True, indent=4)
