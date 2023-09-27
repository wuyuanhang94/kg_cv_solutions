import detectron2
from pathlib import Path
import random, cv2, os
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.evaluator import DatasetEvaluator
setup_logger()

from detectron2.engine import BestCheckpointer
from detectron2.checkpoint import DetectionCheckpointer

dataDir=Path('./input/')
cfg = get_cfg()

cfg.INPUT.MASK_FORMAT='bitmask'

DatasetCatalog.clear()
register_coco_instances('sart_train',{}, '/home/yiw/kg/sart/folds/coco_cell_train_fold5.json', dataDir)
register_coco_instances('sart_val',{},'/home/yiw/kg/sart/folds/coco_cell_valid_fold5.json', dataDir)
metadata = MetadataCatalog.get('sart_train')
train_ds = DatasetCatalog.get('sart_train')

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

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        hooks = super().build_hooks()
        hooks.insert(-1, BestCheckpointer(cfg.TEST.EVAL_PERIOD, 
                                         DetectionCheckpointer(self.model, cfg.OUTPUT_DIR),
                                         "MaP IoU",
                                         "max",
                                         ))
        return hooks


cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("sart_train",)
cfg.DATASETS.TEST = ("sart_val",)
cfg.DATALOADER.NUM_WORKERS = 32

cfg.MODEL.WEIGHTS = "/home/yiw/kg/sart/output_fold5_2602/model_best.pth"

cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.0002
cfg.SOLVER.MAX_ITER = 10000    
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get('sart_train')) // cfg.SOLVER.IMS_PER_BATCH

cfg.INPUT.CROP.ENABLED = True
cfg.INPUT.CROP.SIZE = [0.9, 0.9]

cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "range"
cfg.INPUT.MIN_SIZE_TRAIN = (1200, 1400)
cfg.INPUT.MAX_SIZE_TRAIN = 1400

cfg.INPUT.MIN_SIZE_TEST = 1200
cfg.INPUT.MAX_SIZE_TEST = 1400

cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]
cfg.MODEL.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4.0]]
cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p3', 'p4', 'p5', 'p6']
cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ['p2', 'p3', 'p4', 'p5', 'p6']
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3

cfg.SOLVER.LR_SCHEDULER_NAME = 'WarmupCosineLR'
cfg.SOLVER.WARMUP_ITERS = 100

cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 2.0
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 2.0

with open("cfg.log", "w") as f:
    f.write(str(cfg))

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
