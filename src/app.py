from pathlib import Path
from collections import defaultdict
import csv
import threading
import time

import cv2
from flask import Flask, jsonify, render_template, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO


app = Flask(__name__, template_folder="../interface")
CORS(app)

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "data" / "raw" / "hallway_test.mp4"
TRACKER_CONFIG = BASE_DIR / "botsort_reid.yaml"
ALERT_FILE = BASE_DIR / "alerts.csv"

THRESHOLD_SECONDS = 6.0
COOLDOWN_SECONDS = 5.0

latest_alert = {
    "active": False,
    "message": "",
    "track_id": None,
    "time_sec": None,
    "duration_sec": None,
    "timestamp": 0
}


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 30.0


def init_csv():
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
    with open(ALERT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            video_name,
            track_id,
            round(alert_time, 2),
            round(duration, 2)
        ])


def run_tracking():
    global latest_alert

    if not VIDEO_PATH.exists():
        print(f"Video not found: {VIDEO_PATH}")
        return

    if not TRACKER_CONFIG.exists():
        print(f"Tracker config not found: {TRACKER_CONFIG}")
        return

    init_csv()

    fps = get_fps(VIDEO_PATH)
    threshold_frames = int(THRESHOLD_SECONDS * fps)
    cooldown_frames = int(COOLDOWN_SECONDS * fps)

    print(f"FPS ≈ {fps:.1f} | Alert threshold = {THRESHOLD_SECONDS}s (~{threshold_frames} frames)")

    model = YOLO("yolov8n.pt")
    consecutive = defaultdict(int)
    cooldown = 0
    frame_idx = 0

    results_iter = model.track(
        source=str(VIDEO_PATH),
        tracker=str(TRACKER_CONFIG),
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

                print(f"🚨 ALERT at ~{time_sec:.1f}s: ID {main_id} present for {duration_sec:.1f}s")

                log_alert(VIDEO_PATH.name, main_id, time_sec, duration_sec)

                latest_alert = {
                    "active": True,
                    "message": "Possible follower detected",
                    "track_id": main_id,
                    "time_sec": round(time_sec, 2),
                    "duration_sec": round(duration_sec, 2),
                    "timestamp": time.time()
                }

                cooldown = cooldown_frames

    print("Tracking finished.")


@app.route("/alert", methods=["GET"])
def get_alert():
    return jsonify(latest_alert)


@app.route("/clear_alert", methods=["POST"])
def clear_alert():
    global latest_alert
    latest_alert = {
        "active": False,
        "message": "",
        "track_id": None,
        "time_sec": None,
        "duration_sec": None,
        "timestamp": 0
    }
    return jsonify({"ok": True})

@app.route("/")
def home():
    return render_template("frontend.html")

@app.route("/video")
def serve_video():
    return send_from_directory(VIDEO_PATH.parent, VIDEO_PATH.name)

if __name__ == "__main__":
    tracking_thread = threading.Thread(target=run_tracking, daemon=True)
    tracking_thread.start()

    app.run(host="0.0.0.0", port=5000, debug=True)