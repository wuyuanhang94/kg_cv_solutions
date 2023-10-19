from ultralytics import YOLO

# Load a model
model = YOLO('/home/yiw/kg_yolo/runs/segment/train/weights/best.pt') #m
print(model)

model.train(data='/home/yiw/kg_yolo/yolov8/pseduo.yaml', batch=30, imgsz=1024, epochs=30, warmup_epochs=0, lr0=0.0005)
