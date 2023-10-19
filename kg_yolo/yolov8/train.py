from ultralytics import YOLO

# Load a model
for fold in range(2, 3):
    model = YOLO('/home/yiw/kg_yolo/yolov8/runs/segment/train2_f2_0.380/weights/best.pt')
    model.train(data=f'/home/yiw/kg_yolo/yolov8/fold{fold}.yaml', batch=16, imgsz=1024, epochs=30, lr0=0.0001, mask_ratio=8)
