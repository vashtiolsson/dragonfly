from pathlib import Path
from collections import defaultdict
import csv

import cv2
from ultralytics import YOLO


VIDEO_PATH = Path("data/raw/hallway_test.mp4")

THRESHOLD_SECONDS = 6.0   # alert if same ID present for more than this
COOLDOWN_SECONDS = 5.0    # don't spam alerts constantly

ALERT_FILE = Path("alerts.csv")


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 30.0


def init_csv():
    """Create CSV file with header if it doesn't exist"""
    if not ALERT_FILE.exists():
        with open(ALERT_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "video_name",
                "track_id",
                "alert_time_sec",
                "duration_sec"
            ])


def log_alert(video_name, track_id, alert_time, duration):
    """Append alert to CSV"""
    with open(ALERT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            video_name,
            track_id,
            round(alert_time, 2),
            round(duration, 2)
        ])


def main():

    if not VIDEO_PATH.exists():
        print(f"Video not found: {VIDEO_PATH}")
        print("Put a video in data/raw/ and name it hallway_test.mp4 (or change VIDEO_PATH).")
        return

    init_csv()

    fps = get_fps(VIDEO_PATH)

    threshold_frames = int(THRESHOLD_SECONDS * fps)
    cooldown_frames = int(COOLDOWN_SECONDS * fps)

    print(f"FPS ≈ {fps:.1f} | Alert threshold = {THRESHOLD_SECONDS}s (~{threshold_frames} frames)")

    model = YOLO("yolov8n.pt")

    consecutive = defaultdict(int)

    cooldown = 0
    alert_fired_for_id = None

    frame_idx = 0

    results_iter = model.track(
        source=str(VIDEO_PATH),
        tracker="bytetrack.yaml",
        classes=[0],
        persist=True,
        stream=True,
        save=True
    )

    for r in results_iter:

        frame_idx += 1

        if cooldown > 0:
            cooldown -= 1

        ids_in_frame = set()

        if r.boxes is not None and r.boxes.id is not None:
            ids_in_frame = set(int(x) for x in r.boxes.id.tolist())

        if ids_in_frame:

            for tid in ids_in_frame:
                consecutive[tid] += 1

            for tid in list(consecutive.keys()):
                if tid not in ids_in_frame:
                    consecutive[tid] = 0

            main_id = max(consecutive, key=lambda k: consecutive[k])
            main_frames = consecutive[main_id]

            if main_frames >= threshold_frames and cooldown == 0:

                duration_sec = main_frames / fps
                time_sec = frame_idx / fps

                print(
                    f"🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨 ALERT at ~{time_sec:.1f}s: ID {main_id} present for {duration_sec:.1f}s"
                )

                log_alert(
                    VIDEO_PATH.name,
                    main_id,
                    time_sec,
                    duration_sec
                )

                cooldown = cooldown_frames
                alert_fired_for_id = main_id

        else:
            for tid in list(consecutive.keys()):
                consecutive[tid] = 0

    print("Done. Check runs/track/ for the annotated output video.")
    print("Alerts saved to alerts.csv")


if __name__ == "__main__":
    main()