from ultralytics import YOLO

# Load a model
model = YOLO('/home/yiw/kg_yolo/runs/segment/train11/weights/best.pt')
print(model)

model.train(data='/home/yiw/kg_yolo/yolov8/pseduo_glo.yaml', batch=8, imgsz=768, epochs=6, warmup_epochs=0, lr0=0.00001)
