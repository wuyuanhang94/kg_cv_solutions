import matplotlib.pyplot as plt
import cv2
import json
from pathlib import Path
from pycocotools import _mask
from pycocotools.coco import COCO
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode

# Read LiveCell shsy5y train and val data 
with open('/home/yiw/kg/sart/input/LIVECell_dataset_2021/annotations/LIVECell_single_cells/shsy5y/livecell_shsy5y_train.json') as f:
  data_train = json.loads(f.read())

with open('/home/yiw/kg/sart/input/LIVECell_dataset_2021/annotations/LIVECell_single_cells/shsy5y/livecell_shsy5y_val.json') as f:
  data_val = json.loads(f.read())

categories = [{'name':'shsy5y', 'id':1}, {'name':'astro', 'id':2}, {'name':'cort', 'id':3}]

data_train['categories'] = categories
data_val['categories'] = categories


train_annotations = []
for key in data_train['annotations'].keys():
  rle = _mask.frPoly(data_train['annotations'][key]['segmentation'],520,704)
  data_train['annotations'][key]['segmentation'] = {'size': rle[0]['size'], 'counts':rle[0]['counts'].decode('utf-8')}
  data_train['annotations'][key]['image_id'] = str(data_train['annotations'][key]['image_id'])
  train_annotations.append(data_train['annotations'][key])
data_train['annotations'] = train_annotations