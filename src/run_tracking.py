from pathlib import Path
from ultralytics import YOLO

VIDEO = Path("data/raw/hallway_test.mp4")

def main():
    if not VIDEO.exists():
        print(f" Video not found: {VIDEO}")
        print("Put a video in data/raw/ and name it hallway_test.mp4")
        return

    model = YOLO("yolov8n.pt")  # fast model for prototype

    model.track(
        source=str(VIDEO),
        tracker="bytetrack.yaml",
        classes=[0],       # person only
        persist=True,
        save=True
    )

    print("Done. Output saved under runs/track/")

if __name__ == "__main__":
    main()