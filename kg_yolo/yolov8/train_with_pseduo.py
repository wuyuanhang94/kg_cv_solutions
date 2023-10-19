from ultralytics import YOLO

model_paths = ['/home/yiw/kg_yolo/runs/segment/train9/weights/best.pt']

for fold, mdl in zip(range(4, 5), model_paths[0:]):
    model = YOLO(mdl)
    model.train(data=f'/home/yiw/kg_yolo/yolov8/fold{fold}_with_plabel.yaml', batch=24, imgsz=768, epochs=88, lr0=0.0001, mask_ratio=1)
