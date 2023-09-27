import torch, torchvision
import pandas as pd
import numpy as np
import pandas as pd 
from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm # progress bar
from datetime import datetime
import time
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util

from pycocotools.coco import COCO
import os, json, cv2, random
import skimage.io as io
import copy
from pathlib import Path
from typing import Optional
import pickle

from tqdm import tqdm
import itertools

import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from glob import glob
import numba
from numba import jit

import warnings
warnings.filterwarnings('ignore') #Ignore "future" warnings and Data-Frame-Slicing warnings.

# detectron2
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation.evaluator import DatasetEvaluator

from detectron2.engine import BestCheckpointer
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

setup_logger()

imgdir = "/home/yiw/kg/sart/input/train"
debug = False
split_mode = "valid20" # 20% for validation

train_df = pd.read_csv('/home/yiw/kg/sart/folds/train_fold1.csv')
thing_classes=train_df['cell_type'].unique()
thing_classes=['shsy5y','astro','cort']
thing_classes_to_id = {class_name: index for index, class_name in enumerate(thing_classes)}

train_df["cell_type_int"]=train_df["cell_type"]
train_df.loc[train_df.cell_type_int == "shsy5y", 'cell_type_int'] = 0
train_df.loc[train_df.cell_type_int == "astro", 'cell_type_int'] = 1
train_df.loc[train_df.cell_type_int == "cort", 'cell_type_int'] = 2

train_meta=train_df.copy(deep=True)
train_meta=train_meta.drop_duplicates(subset=['id'])
train_meta=train_meta.reset_index(drop=True)
train_meta=train_meta[["id","width","height"]]

def rle_polygon_coco(mask_rle,category_id,image_id,annotation_id):
    ground_truth_binary_mask =mask_rle
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)
    bbox1=ground_truth_bounding_box.tolist()
    boox2=[bbox1[0],bbox1[1],bbox1[0]+bbox1[2],bbox1[1]+bbox1[3]]
    s=1
    if ((bbox1[2]<=0)or(bbox1[3]<=0) or (bbox1[0]+bbox1[2]>703) or (bbox1[1]+bbox1[3]>519)):
        s=0
    annotation = {
            "iscrowd": 0, # 0 polygon    1 rle
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": category_id,
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            'image_id': image_id,
            'id': annotation_id,
            "bbox_mode": BoxMode.XYWH_ABS,
        }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        if len(segmentation) <7:
            s=0
        annotation["segmentation"].append(segmentation)
    return annotation,s

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height, width, channels) of array to return 
    color: color for the mask
    Returns numpy array (mask)

    '''
    s = mask_rle.split()
    
    starts = list(map(lambda x: int(x) - 1, s[0::2]))
    lengths = list(map(int, s[1::2]))
    
    ends = [x + y for x, y in zip(starts, lengths)]
    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
            
    for start, end in zip(starts, ends):
        img[start : end] = color
    
    return img.reshape(shape)

def coco_structure(train_df):
    cat_ids = {name:id+1 for id, name in enumerate(train_df.cell_type.unique())}    
    cats =[{'name':name, 'id':id} for name,id in cat_ids.items()]
    images = [{'id':id, 'width':row.width, 'height':row.height, 'file_name':f'train/{id}.png'} for id,row in train_df.groupby('id').agg('first').iterrows()]
    annotations=[]
    for idx, row in tqdm(train_df.iterrows()):
        mk = rle_decode(row.annotation, (row.height, row.width))
        ys, xs = np.where(mk)
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        enc =binary_mask_to_rle(mk)
        seg = {
            'segmentation':enc, 
            'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)],
            'area': int(np.sum(mk)),
            'image_id':row.id, 
            'category_id':cat_ids[row.cell_type], 
            'iscrowd':0, 
            'id':idx
        }
        annotations.append(seg)
    return {'categories':cats, 'images':images,'annotations':annotations}

import json
import numpy as np
from pycocotools import mask
from skimage import measure

debug=False
def get_Cell_data_dicts(
    imgdir: Path,
    train_df: pd.DataFrame,
    train_meta: pd.DataFrame,
    use_cache: bool = True,
    target_indices: Optional[np.ndarray] = None,
    debug: bool = False,
    data_type:str="train"
):
    cache_path = Path(".") / f"dataset_dicts_cache_{data_type}.pkl"
    if not use_cache or not cache_path.exists():
        print("Creating data...")
        dataset_dicts = []

        for index, train_meta_row in tqdm(train_meta.iterrows(), total=len(train_meta)):
            record = {}
            image_id, width,height = train_meta_row.values
            filename = str(f'{imgdir}/{image_id}.png')
            record["file_name"] = filename
            record["image_id"] = index
            record["width"] = width
            record["height"] = height
            objs = []
            for index2, row in train_df.query("id == @image_id").iterrows():                       
                ann = row["annotation"]
                if len(ann)>25:
                    mk = np.zeros((520, 704, 1))
                    mk = rle_decode(ann, shape=(520, 704, 1), color=1)
                    mkk = mk[:, :, 0]
                    mkk=np.array(mkk, dtype=np.uint8)
                    seg,s=rle_polygon_coco(mkk,row["cell_type_int"],index,index2)
                    if s==1:
                        objs.append(seg)
            record["annotations"] = objs
            dataset_dicts.append(record)
        with open(cache_path, mode="wb") as f:
            pickle.dump(dataset_dicts, f)

    print(f"Load from cache {cache_path}")
    with open(cache_path, mode="rb") as f:
        dataset_dicts = pickle.load(f)
    if target_indices is not None:
        dataset_dicts = [dataset_dicts[i] for i in target_indices]
    return dataset_dicts
    
Data_Resister_training="Cell_data_train";
Data_Resister_valid="Cell_data_valid";

if split_mode == "valid20":
    DatasetCatalog.register(
        Data_Resister_training,
        lambda: get_Cell_data_dicts(
            imgdir,
            train_df,
            train_meta,
            target_indices=None,
            debug=debug,
            data_type="train"
        ),
    )
    MetadataCatalog.get(Data_Resister_training).set(thing_classes=thing_classes)
    
    dataDir=Path('./input/')
    register_coco_instances(Data_Resister_valid,{},'/home/yiw/kg/sart/folds/coco_cell_valid_fold1.json', dataDir)

    dataset_dicts_train = DatasetCatalog.get(Data_Resister_training)
    metadata_dicts_train = MetadataCatalog.get(Data_Resister_training)

class_shsy5y=0
class_astro=0
class_cort=0
for d in dataset_dicts_train:
    for i in d['annotations']:
        if i["category_id"]==0:
            class_shsy5y=class_shsy5y+1
        elif i["category_id"]==1:
            class_astro=class_astro+1
        elif i["category_id"]==2:
            class_cort=class_cort+1
print("distribution of classes in training")
print("class_shsy5y=",class_shsy5y,"class_astro=",class_astro,"class_cort=",class_cort)

def custom_mapper(dataset_dict):
    
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
            T.RandomBrightness(0.8, 1.2),
            # T.RandomContrast(0.8, 1.2),
            # T.RandomSaturation(0.8, 1.2),
            # T.RandomLighting(0.8),
            # T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            # T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

def score(pred, targ):
    pred_masks = pred['instances'].pred_masks.cpu().numpy()
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ))
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)

class MAPIOUEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in dataset_dicts}
            
    def reset(self):
        self.scores = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out['instances']) == 0:
                self.scores.append(0)    
            else:
                targ = self.annotations_cache[inp['image_id']]
                self.scores.append(score(out, targ))

    def evaluate(self):
        return {"MaP IoU": np.mean(self.scores)}


class AugTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    def build_hooks(self):
        cfg = self.cfg.clone()
        hooks = super().build_hooks()
        hooks.insert(-1, BestCheckpointer(cfg.TEST.EVAL_PERIOD, 
                                         DetectionCheckpointer(self.model, cfg.OUTPUT_DIR),
                                         "MaP IoU",
                                         "max",
                                         ))
        return hooks


cfg = get_cfg()

config_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" 

cfg.merge_from_file(model_zoo.get_config_file(config_name))


cfg.DATASETS.TRAIN = (Data_Resister_training,)
if split_mode == "all_train":
    cfg.DATASETS.TEST = ()
else:
    cfg.DATASETS.TEST = (Data_Resister_valid,)

# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = "/home/yiw/kg/sart/output/model_best.pth"

cfg.DATALOADER.NUM_WORKERS = 32
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  # 64 is slower but more accurate (128 faster but less accurate)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
cfg.SOLVER.IMS_PER_BATCH = 8 #(2 is per defaults)
cfg.SOLVER.BASE_LR = 0.0001 #(quite high base learning rate but should drop)
cfg.SOLVER.WARMUP_ITERS = 0 #How many iterations to go from 0 to reach base LR
cfg.SOLVER.MAX_ITER = 10000 #Maximum of iterations 1
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
cfg.INPUT.MASK_FORMAT='polygon'
# cfg.TEST.EVAL_PERIOD = 5 * len(DatasetCatalog.get(Data_Resister_training)) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch
cfg.TEST.EVAL_PERIOD = 2 * len(DatasetCatalog.get(Data_Resister_training)) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = AugTrainer(cfg) # with  data augmentation  
trainer.resume_or_load(resume=False)
trainer.train()
