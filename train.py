from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

model.train(data='/data/mreza/CaveSegmentation/code/Yolo/config.yaml', epochs=300, imgsz=640, batch = 16, device = 1)