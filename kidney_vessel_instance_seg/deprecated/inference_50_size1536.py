import os
import cv2
import json
import glob
import time
import numpy as np
import pandas as pd
import torch
import detectron2
from tqdm.auto import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data import DatasetCatalog, build_detection_test_loader
import pycocotools.mask as mask_util
from PIL import Image
import matplotlib.pyplot as plt
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.modeling import GeneralizedRCNNWithTTA as _GeneralizedRCNNWithTTA
from ensemble_boxes import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('GPU is available')
else:
    DEVICE = torch.device('cpu')
    print('CPU is used')
print('detectron ver:', detectron2.__version__)

train = glob.glob("/home/yiw/kg/input/train/*")
test = glob.glob("/home/yiw/kg/input/test/*")

annotations = {}

# Open the annotations file
with open('/home/yiw/kg/input/polygons.jsonl', 'r') as f:
    for line in f:
        annotation = json.loads(line)
        image_id = annotation['id']
        image_annotations = annotation['annotations']
        annotations[image_id] = image_annotations

image_map = {impath.split('/')[-1].split('.')[0]: impath for impath in train}
unlabelled_id = image_map.keys() - annotations.keys()

config_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
mdl_path = "/home/yiw/kg/models_1536"
DATA_PATH = "/home/yiw/kg/input/"
MODELS = []
BEST_MODELS =[]
THSS = []
ID_TEST = 0
SUBM_PATH = f'{DATA_PATH}/train'
SINGLE_MODE = False
NMS = True
IOU_TH = .6

THRESHOLDS = [.25, .99]
MIN_PIXELS = [1, 60]

mdl_path = "/home/yiw/kg/models_1536"
best_model = ["model_best_1536_fold0.pth", "model_best_1536_fold1.pth", "model_best_1536_fold2.pth", "model_best_1536_fold3.pth", "model_best_1536_fold4.pth"]

for b_m in best_model:
    model_name=b_m
    model_ths=THRESHOLDS
    BEST_MODELS.append(model_name)
    THSS.append(model_ths)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_name))
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = f'{mdl_path}/{model_name}'

    cfg.INPUT.MIN_SIZE_TEST = 1536
    cfg.INPUT.MAX_SIZE_TEST = 1536

    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    MODELS.append(DefaultPredictor(cfg))
print(MODELS)

def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) 
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def pred_masks(file_name, path, model, ths, min_pixels):
    img = cv2.imread(f'{path}/{file_name}')
    output = model(img)
    pred_classes = output['instances'].pred_classes.cpu().numpy().tolist()
    pred_class = max(set(pred_classes), key=pred_classes.count)
    take = output['instances'].scores >= ths[pred_class]
    pred_masks = output['instances'].pred_masks[take]
    pred_masks = pred_masks.cpu().numpy()
    result = []
    used = np.zeros(img.shape[:2], dtype=int) 
    for i, mask in enumerate(pred_masks):
        mask = mask * (1 - used)
        if mask.sum() >= min_pixels[pred_class]:
            used += mask
            result.append(rle_encode(mask))
    return result

def ensemble_preds(file_name, path, models, ths):
    img = cv2.imread(f'{path}/{file_name}')
    
    outputs = []
    
    classes = []
    scores = []
    bboxes = []
    masks = []
    
    pred_classes_gros = []
    
    for i, model in enumerate(models):
        output = model(img)
        outputs.append(output)
        
        pred_classes = output['instances'].pred_classes.cpu().numpy().tolist()
        if len(pred_classes) < 1: continue
        pred_class = max(set(pred_classes), key=pred_classes.count)
        #print(f"old model {i} predict class {pred_class}")
        pred_classes_gros.append(pred_class)
    
    if len(pred_classes_gros) < 1: return []
    pred_class_final = max(set(pred_classes_gros), key=pred_classes_gros.count)
    
    for c, output in zip(pred_classes_gros, outputs):
        if c != pred_class_final:
            continue
        take = output['instances'].scores >= ths[i][pred_class_final]
        classes.extend(output['instances'].pred_classes[take].cpu().numpy().tolist())
        scores.extend(output['instances'].scores[take].cpu().numpy().tolist())
        bboxes.extend(output['instances'].pred_boxes[take].tensor.cpu().numpy().tolist())
        masks.extend(output['instances'].pred_masks[take].cpu().numpy())

    assert len(classes) == len(masks) , 'ensemble lenght mismatch'
    return classes, scores, bboxes, masks

def nms_predictions(classes, scores, bboxes, masks, 
                    iou_th=.5, shape=(512, 512)):
    he, wd = shape[0], shape[1]
    boxes_list = [[[x[0] / wd, x[1] / he, x[2] / wd, x[3] / he] for x in bboxes]]
    scores_list = [[x for x in scores]]
    classes_list = [[x for x in classes]]
    nms_bboxes, nms_scores, nms_classes = non_maximum_weighted(
        boxes_list, 
        scores_list, 
        classes_list, 
        weights=None,
        iou_thr=IOU_TH,
        skip_box_thr=0.0001,
    )
    nms_masks = []
    for s in nms_scores:
        nms_masks.append(masks[scores.index(s)])
    if len(masks) < 1:
        return nms_classes, nms_scores, nms_masks
    nms_scores, nms_classes, nms_masks, nms_bboxes = zip(*sorted(zip(nms_scores, nms_classes, nms_masks, nms_bboxes), reverse=True))
    return nms_classes, nms_scores, nms_masks, nms_bboxes

def ensemble_pred_masks(masks, classes, min_pixels, shape=(512, 512)):
    result = []
    #pred_class = max(set(classes), key=classes.count)
    pred_class = int(max(set(classes), key=classes.count).item())
    used = np.zeros(shape, dtype=int) 
    for i, mask in enumerate(masks):
        mask = mask * (1 - used)
        if mask.sum() >= min_pixels[pred_class]:
            used += mask
            result.append(rle_encode(mask))
    return result

test_names = [f"{image_id}.tif" for image_id in unlabelled_id]
print('test images:', len(test_names))

from tqdm import tqdm
import json, itertools
idx = 0
cats =[{'name': "blood_vessel", 'id':1}]
images = [{'id':image_id, 'width': 512, 'height': 512, 'file_name': image_map[image_id]} for image_id in annotations.keys()]
coco_annotations = []

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

for test_name in tqdm(test_names):
    img = cv2.imread(f'{SUBM_PATH}/{test_name}')
    h, w, _ = img.shape
    if SINGLE_MODE:
        encoded_masks = pred_masks(
            test_name, 
            path=SUBM_PATH, 
            model=MODELS[0],
            ths=THSS[0],
            min_pixels=MIN_PIXELS
        )
    else:
        classes, scores, bboxes, masks = ensemble_preds(
            file_name=test_name, 
            path=SUBM_PATH, 
            models=MODELS, 
            ths=THSS
        )
        if NMS:
            classes, scores, masks, bboxes = nms_predictions(
            classes, 
            scores, 
            bboxes, 
            masks, 
            iou_th=IOU_TH
        )
        result = Instances([512, 512])
        result.pred_boxes = Boxes(torch.from_numpy(bboxes).to('cuda'))
        result.scores = torch.from_numpy(scores).to('cuda')
        result.pred_classes = torch.from_numpy(classes).to('cuda')

        for clss, mk in zip(classes, masks):
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
            coco_annotations.append(seg)
            idx += 1

images = [{'id':image_id, 'width': 512, 'height': 512, 'file_name': image_map[image_id]} for image_id in unlabelled_id]
final_coco_annos = {'categories':cats, 'images':images,'annotations': coco_annotations}

with open(f'pseduo_label.json', 'w', encoding='utf-8') as f:
    json.dump(final_coco_annos, f, ensure_ascii=True, indent=4)
