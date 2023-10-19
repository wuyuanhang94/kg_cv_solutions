import json
from pathlib import Path
import os
import shutil
import cv2
import itertools
import numpy as np
from typing import List, Dict
from sklearn.model_selection import KFold

DATA_DIR = Path('/home/yiw/kg_yolo/input')
id_dict = {'blood_vessel': 0, 'glomerulus': 1, 'unsure': 2}

# Function to copy images and transform labels to 
# coco formatted .txt files
def tile_to_coco(tile: List[Dict], output_folder: Path):
    tile_id = tile['id']    
    shutil.copyfile(DATA_DIR / f'train/{tile_id}.tif', output_folder / f'{tile_id}.tif')

    with open(output_folder / f'{tile_id}.txt', 'w') as text_file:
        for annotation in tile['annotations']:
            if annotation['type'] != "blood_vessel": continue
            class_id = id_dict[annotation['type']]
            flat_mask_polygon = list(itertools.chain(*annotation['coordinates'][0]))
            array = np.array(flat_mask_polygon)/512.
            text_file.write(f'{class_id} {" ".join(map(str, array))}\n')

with open('/home/yiw/kg_yolo/input/polygons.jsonl', 'r') as json_file:
    json_list = list(json_file)

tiles_dicts = []
for json_str in json_list:
    tiles_dicts.append(json.loads(json_str))
tiles_dicts = np.array(tiles_dicts)

os.makedirs("folds", exist_ok=True)
folds = KFold(n_splits=5, random_state=23, shuffle=True).split(tiles_dicts)

for fold, (train_idx, val_idx) in enumerate(folds):
    train_dicts, valid_dicts = tiles_dicts[train_idx], tiles_dicts[val_idx]
    os.makedirs(f'folds/train{fold}', exist_ok=True)
    os.makedirs(f'folds/valid{fold}', exist_ok=True)

    for train_dict in train_dicts: 
        tile_to_coco(train_dict, Path(f'/home/yiw/kg_yolo/yolov8/folds/train{fold}/'))
    for valid_dict in valid_dicts: 
        tile_to_coco(valid_dict, Path(f'/home/yiw/kg_yolo/yolov8/folds/valid{fold}/'))

    yaml_text = f"""
    train: /home/yiw/kg_yolo/yolov8/folds/train{fold}/
    val: /home/yiw/kg_yolo/yolov8/folds/valid{fold}/

    nc: 1
    names: ['blood_vessel']
    """

    with open(f'/home/yiw/kg_yolo/yolov8/fold{fold}.yaml', 'w') as text_file:
        text_file.write(yaml_text)
