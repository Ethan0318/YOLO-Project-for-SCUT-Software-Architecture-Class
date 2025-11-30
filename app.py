# 标准库：时间与路径
import time
from pathlib import Path

# 第三方库：OpenCV、Flask、Ultralytics YOLO、Werkzeug
import cv2
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# 项目目录与路径常量
BASE_DIR = Path(__file__).resolve().parent
# 允许上传的图片扩展名
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
# 上传与结果输出目录
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "result"
# YOLO 模型文件路径
MODEL_PATH = BASE_DIR / "models" / "yolov8s.pt"

# Flask 应用初始化与配置
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["RESULT_FOLDER"] = str(RESULT_FOLDER)
# 限制上传文件大小（200MB）
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB 上传限制

# 确保上传与结果目录存在
for folder in (UPLOAD_FOLDER, RESULT_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

# 全局模型句柄（懒加载）
model = None


# 加载 YOLO 模型（懒加载）
def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        model = YOLO(str(MODEL_PATH))
    return model


# 判断文件扩展名是否受支持
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# 转换为相对路径（相对于项目根目录），便于前端引用
def to_relative(path: Path) -> str:
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


# 首页：渲染上传页面
@app.route("/")
def index():
    return render_template("index.html")


# 检测接口：接收上传图片并根据策略进行处理
@app.route("/detect", methods=["POST"])
def detect():
    total_start = time.time()
    # 校验是否包含文件
    if "file" not in request.files:
        return jsonify({"error": "file field missing"}), 400

    uploaded = request.files["file"]
    # 选择处理策略（A/B/C），默认 A
    strategy = request.form.get("strategy", "A").upper()

    # 基本校验
    if uploaded.filename == "":
        return jsonify({"error": "empty filename"}), 400
    if not allowed_file(uploaded.filename):
        return jsonify({"error": "unsupported file type"}), 400

    # 保存上传文件
    filename = secure_filename(uploaded.filename)
    save_path = UPLOAD_FOLDER / filename

    save_start = time.time()
    uploaded.save(save_path)
    save_io_time = time.time() - save_start

    # 读取图片用于推理
    decode_start = time.time()
    img = cv2.imread(str(save_path))
    decode_time = time.time() - decode_start
    # 服务器接收阶段时间（IO + 解码）
    server_recv_pre = save_io_time + decode_time


    # C 策略：仅客户端处理，不进行服务器推理
    if strategy == "C":
        return jsonify(
            {
                "strategy": "C",
                "msg": "client-side-only, no server inference",
            }
        )

    # 加载模型（懒加载）
    try:
        mdl = load_model()
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": str(exc)}), 500

    if strategy == "A":
        # 服务器推理并返回渲染后的图片
        infer_start = time.time()
        results = mdl(img, verbose=False)[0]
        infer_time = time.time() - infer_start

        # 绘制检测结果到图像
        plot_start = time.time()
        annotated = results.plot()
        post_render_time = time.time() - plot_start

        detected_name = f"detected_{Path(filename).stem}.jpg"
        detected_path = RESULT_FOLDER / detected_name
        # 编码并保存渲染结果
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
        # 服务器推理并返回检测框数据（不返回渲染图）
        infer_start = time.time()
        results = mdl(img, verbose=False)[0]
        infer_time = time.time() - infer_start

        # 组装检测框结果
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

    # 未知策略
    return jsonify({"error": "unknown strategy"}), 400


# 在开发模式下运行 Flask 服务
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
