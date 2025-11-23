🚀 YOLOv8 混合分区 Web 系统

基于 Flask + YOLOv8 的目标检测演示系统，支持三种计算分区策略，实现 端云协同（Hybrid Partitioning） 的性能对比。

🧩 功能与三种策略
Strategy A — 纯服务器端推理与画框（Server Only）

前端上传图片/视频

服务器执行：推理 + 后处理（画框）

返回带标注图像或视频

Strategy B — 服务器推理 · 客户端画框（Server Inference + Client Postprocess）

服务器执行 YOLOv8 推理

服务器返回 bounding boxes 数组

前端使用 Canvas 绘制检测框（仅图片）

Strategy C — 浏览器端推理（Client Only · ONNX Runtime Web）

纯前端执行：预处理 + ONNX 推理 + NMS + 画框

无需上传图像

仅支持图片（ONNX Web 推理）

📁 项目结构
project/
│── app.py
│── models/
│    └── yolov8n.pt
│── templates/
│    └── index.html
│── static/
│    ├── uploads/
│    ├── result/
│    ├── css/
│    │    └── style.css
│    └── js/
│         ├── main.js
│         └── yolov8n.onnx      # Strategy C 使用
│── README.md

🔧 环境准备
1. 创建 Conda 环境
conda create -n yoloenv python=3.10 -y
conda activate yoloenv

2. 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install flask ultralytics opencv-python onnxruntime onnx

📥 模型准备（）
✔ 1. 下载 YOLOv8n.pt（用于 Strategy A/B）

以下代码自动下载模型，无需运行整个系统。

from ultralytics import YOLO
YOLO("yolov8n.pt")


然后将 yolov8n.pt 放入：

models/yolov8n.pt

✔ 2.导出 ONNX 模型（用于 Strategy C）

使用 ultralytics 导出：

yolo export model=yolov8n.pt format=onnx opset=12

导出后将 yolov8n.onnx 放入：

static/js/yolov8n.onnx

或使用 Python（推荐）：

python export_onnx.py


▶️ 启动项目
python app.py


打开浏览器访问：

http://localhost:5000/

💡 使用说明

上传图片或视频（支持：.jpg / .jpeg / .png / .mp4）

选择策略：

A：服务器推理 + 服务器画框

B：服务器推理 → 客户端画框（仅图片）

C：浏览器 ONNX 推理（仅图片）

点击 开始检测

查看：

原图 / 结果图对比展示

下载按钮

各策略独立的性能统计表

系统会自动展示上传时间、推理时间、后处理时间、整体耗时等关键性能指标。

📊 性能指标
指标	Strategy A	Strategy B	Strategy C
上传时间	✔	✔	N/A
服务器推理时间	✔	✔	N/A
客户端后处理时间	N/A	✔	✔
整体耗时	✔	✔	✔
🔐 安全与限制

白名单扩展名：jpg, jpeg, png, mp4

使用 secure_filename 防止路径攻击

最大上传大小：200 MB（可在 app.py 调整）

Strategy B/C 暂不支持视频（代码已限制）

🛠️ 开发提示

上传文件保存至：static/uploads/

检测结果保存至：static/result/

若部署到生产环境：

关闭 debug=True

建议使用 Gunicorn + Nginx 或其他 WSGI
