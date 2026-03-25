from pathlib import Path
from collections import defaultdict
import csv
import time

import cv2
from flask import Flask, jsonify, render_template, Response
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__, template_folder="../interface")
CORS(app)

BASE_DIR = Path(__file__).resolve().parent.parent
TRACKER_CONFIG = BASE_DIR / "botsort_reid.yaml"
ALERT_FILE = BASE_DIR / "alerts.csv"

THRESHOLD_SECONDS = 12.0
COOLDOWN_SECONDS = 5.0

latest_alert = {
    "active": False,
    "message": "",
    "track_id": None,
    "time_sec": None,
    "duration_sec": None,
    "timestamp": 0
}

model = YOLO("yolov8n.pt")

consecutive = defaultdict(int)
cooldown_frames_left = 0
track_start_time = time.time()


def init_csv():
    if not ALERT_FILE.exists():
        with open(ALERT_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "source_name",
                "track_id",
                "alert_time_sec",
                "duration_sec"
            ])


def log_alert(source_name, track_id, alert_time, duration):
    with open(ALERT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            source_name,
            track_id,
            round(alert_time, 2),
            round(duration, 2)
        ])


@app.route("/")
def home():
    return render_template("frontend.html")


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


def generate_camera_stream():
    global latest_alert, cooldown_frames_left, consecutive, track_start_time

    init_csv()

    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    threshold_frames = int(THRESHOLD_SECONDS * fps)
    cooldown_frames = int(COOLDOWN_SECONDS * fps)

    print(f"Webcam FPS ≈ {fps:.1f}")
    print(f"Alert threshold = {THRESHOLD_SECONDS}s (~{threshold_frames} frames)")

    while True:
        success, frame = cap.read()
        if not success:
            break

        if cooldown_frames_left > 0:
            cooldown_frames_left -= 1

        # Run tracking on this frame
        results = model.track(
            source=frame,
            tracker=str(TRACKER_CONFIG),
            classes=[0],      # person class
            persist=True,
            verbose=False
        )

        result = results[0]
        annotated_frame = result.plot()

        ids_in_frame = set()
        if result.boxes is not None and result.boxes.id is not None:
            ids_in_frame = set(int(x) for x in result.boxes.id.tolist())

        if ids_in_frame:
            for tid in ids_in_frame:
                consecutive[tid] += 1

            for tid in list(consecutive.keys()):
                if tid not in ids_in_frame:
                    consecutive[tid] = 0

            main_id = max(consecutive, key=lambda k: consecutive[k])
            main_frames = consecutive[main_id]

            if main_frames >= threshold_frames and cooldown_frames_left == 0:
                duration_sec = main_frames / fps
                time_sec = time.time() - track_start_time

                print(f"🚨 ALERT at ~{time_sec:.1f}s: ID {main_id} present for {duration_sec:.1f}s")

                log_alert("webcam", main_id, time_sec, duration_sec)

                latest_alert = {
                    "active": True,
                    "message": "Possible follower detected",
                    "track_id": main_id,
                    "time_sec": round(time_sec, 2),
                    "duration_sec": round(duration_sec, 2),
                    "timestamp": time.time()
                }

                cooldown_frames_left = cooldown_frames

        # Encode frame as JPEG
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


@app.route("/camera_feed")
def camera_feed():
    return Response(
        generate_camera_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)