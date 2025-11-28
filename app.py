import time
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "result"
MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["RESULT_FOLDER"] = str(RESULT_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB upload cap

for folder in (UPLOAD_FOLDER, RESULT_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

model = None


def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
    return model


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def to_relative(path: Path) -> str:
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    total_start = time.time()
    if "file" not in request.files:
        return jsonify({"error": "file field missing"}), 400

    uploaded = request.files["file"]
    strategy = request.form.get("strategy", "A").upper()

    if uploaded.filename == "":
        return jsonify({"error": "empty filename"}), 400
    if not allowed_file(uploaded.filename):
        return jsonify({"error": "unsupported file type"}), 400

    filename = secure_filename(uploaded.filename)
    save_path = UPLOAD_FOLDER / filename

    save_start = time.time()
    uploaded.save(save_path)
    save_io_time = time.time() - save_start

    decode_start = time.time()
    img = cv2.imread(str(save_path))
    decode_time = time.time() - decode_start
    server_recv_pre = save_io_time + decode_time


    if strategy == "C":
        return jsonify(
            {
                "strategy": "C",
                "msg": "client-side-only, no server inference",
            }
        )

    try:
        mdl = load_model()
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": str(exc)}), 500

    if strategy == "A":
        infer_start = time.time()
        results = mdl(img, verbose=False)[0]
        infer_time = time.time() - infer_start

        plot_start = time.time()
        annotated = results.plot()
        post_render_time = time.time() - plot_start

        detected_name = f"detected_{Path(filename).stem}.jpg"
        detected_path = RESULT_FOLDER / detected_name
        encode_start = time.time()
        cv2.imwrite(str(detected_path), annotated)
        encode_time = time.time() - encode_start

        resp_prep_start = time.time()
        payload = {
            "strategy": "A",
            "original": to_relative(save_path),
            "detected": to_relative(detected_path),
            "timings": {
                "server_recv_pre": round(server_recv_pre, 3),
                "server_infer": round(infer_time, 3),
                "server_post": round(post_render_time + encode_time + (time.time() - resp_prep_start), 3),
            },
        }
        return jsonify(payload)

    if strategy == "B":
        infer_start = time.time()
        results = mdl(img, verbose=False)[0]
        infer_time = time.time() - infer_start

        assemble_start = time.time()
        boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                boxes.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cls": int(box.cls[0]),
                        "conf": float(box.conf[0]),
                    }
                )
        assemble_time = time.time() - assemble_start

        resp_prep_start = time.time()
        payload = {
            "strategy": "B",
            "original": to_relative(save_path),
            "boxes": boxes,
            "timings": {
                "server_recv_pre": round(server_recv_pre, 3),
                "server_infer": round(infer_time, 3),
                "server_post": round(assemble_time + (time.time() - resp_prep_start), 3),
            },
        }
        return jsonify(payload)

    return jsonify({"error": "unknown strategy"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
