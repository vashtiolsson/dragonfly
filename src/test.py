from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.track(source="data/raw/hallway_test.mp4", 
            tracker="bytetrack.yaml", 
            classes=[0], persist=True, 
            show=True)