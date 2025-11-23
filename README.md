# YOLOv8 混合分区 Web 系统

基于 Flask + YOLOv8 的目标检测演示，支持三种计算分区策略：
- Strategy A：纯服务器端推理与画框
- Strategy B：服务器推理，客户端画框
- Strategy C：纯前端（浏览器 ONNX/Web 推理与画框）

## 目录结构
project/
│── app.py
│── models/
│ └── yolov8n.pt
│── templates/
│ └── index.html
│── static/
│ ├── uploads/
│ ├── result/
│ ├── css/
│ └── js/
│ └── main.js
└── README.md



## 环境准备
- Python 3.9+
- 依赖安装：
  ```bash
 conda create -n yoloenv python=3.10 -y
 conda activate yoloenv
 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
 pip install flask ultralytics opencv-python onnxruntime onnx
'''
注意下面的下载是为了获取模型文件（仅作展示，不用运行）
下载权重：
bash

python - <<'PY'
from ultralytics import YOLO
YOLO('yolov8n.pt')
PY

或者
bash
python expert_onnx.py


将 `yolov8n.pt` 放到 `models/`。
- （可选）Strategy C 需要浏览器侧 ONNX：
```bash
yolo export model=yolov8n.pt format=onnx opset=12
将导出的 yolov8n.onnx 放到 static/js/，前端默认加载 /static/js/yolov8n.onnx。

运行
bash

python app.py
浏览器访问 http://localhost:5000/。

使用说明
选择图片/视频（jpg/jpeg/png/mp4）。
选择策略：
A：服务器推理+画框，返回带框图/视频。
B：服务器仅返回 boxes，前端 canvas 画框（当前代码只处理图片）。
C：浏览器 ONNX 推理 + 前端 NMS/画框（当前代码只处理图片）。
点击“开始检测”，查看状态、进度条与性能表。
各策略有独立结果面板与下载链接，不会互相覆盖。
性能指标
上传时间：A/B 显示；C 不适用。
服务器推理时间：A/B 显示；C 不适用。
客户端后处理时间：B/C 显示。
整体耗时：A/B/C 均显示。
安全与限制
上传白名单：jpg、jpeg、png、mp4；使用 secure_filename。
默认最大上传 200MB（app.py 中可调整）。
Strategy B/C 当前不支持视频后处理；需要自行扩展后处理逻辑可参考 canvas 渲染部分。
开发提示
静态文件写入路径：static/uploads/、static/result/（需存在或自动创建）。
若部署到生产环境，请关闭 debug=True，并放置在 WSGI/反向代理后。
前端 ONNX 推理使用 onnxruntime-web；可根据环境选择 WASM/WebGL 执行提供器。

