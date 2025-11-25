# 基于YOLOv8的图片检测系统

基于 Flask + YOLOv8 的目标检测演示，支持服务器 / 浏览器混合分区推理。三种策略可切换并对比耗时：

- **Strategy A**：纯服务器推理与画框
- **Strategy B**：服务器推理，客户端画框
- **Strategy C**：纯前端（浏览器 ONNX/Web 推理与画框）

## ver-1.0 更新说明

- 前端界面略微调整，删去了一些多余的文字
- 修复了策略c检测时的多检测框问题
- 修复了策略b，c的检测框过细、标签字号太小的问题（策略a仍然使用默认的检测框设置）
- 调整了性能检测模块的指标，改为请求准备与上传、客户端下载与解析、整体耗时三类，这三个指标指代明确，易于计算，且能够反映三种策略的核心差别
- 将原来策略a的视频检测功能删除了

## 目录结构

```
project/
├─ app.py
├─ models/
│   └─ yolov8n.pt
├─ templates/
│   └─ index.html
├─ static/
│   ├─ uploads/
│   ├─ result/
│   ├─ css/
│   └─ js/
│       └─ main.js
└─ README.md
```

## 环境与依赖

- Python 3.9+
- 创建环境：
  ```bash
  conda create -n yoloenv python=3.10 -y
  conda activate yoloenv

  ```
- 安装依赖：
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install flask ultralytics opencv-python onnxruntime onnx
  ```
  以下操作不需要在bash中操作，仅供参考模型参数怎么来：
- 下载模型权重：
  ```bash
  python
  from ultralytics import YOLO
  YOLO("yolov8n.pt")
  exit()
  ```
  将生成的 `yolov8n.pt` 放到 `models/`。
- 浏览器侧 ONNX（供 Strategy C 使用）：
  ```bash
  yolo export model=yolov8n.pt format=onnx opset=12
  ```
  把导出的 `yolov8n.onnx` 放到 `static/js/`，前端默认加载 `/static/js/yolov8n.onnx`。

- 推荐下载方式（使用py程序）：
  ```bash
  python export_onnx.py
  ```

## 运行

```bash
python app.py
```

启动后访问：`http://localhost:5000/`

## 使用流程

1. 选择文件：支持 `jpg / jpeg / png`。
2. 选择策略：
   - **A**：服务器推理+画框，返回带框图片。
   - **B**：服务器仅返回 boxes，前端 canvas 画框。
   - **C**：浏览器 ONNX 推理 + 前端 NMS/画框。
3. 点击“开始检测”，查看状态、进度与性能表。
4. 各策略有独立结果面板和下载链接，依次运行 A→B→C 也不会互相覆盖。

## 性能指标

- 请求准备与上传：文件读取/压缩/序列化、HTTP/TLS 握手、请求排队、上传正文完成（A/B 显示，C 不适用）。
- 客户端下载与解析：从首字节到尾字节的下载、图片/JSON 解码（A/B 显示，C 不适用）
- 整体耗时：用户发起操作→结果可见的关键路径时间，A/B/C 均显示。

## 安全与限制

- 上传白名单：`jpg`、`jpeg`、`png`；使用 `secure_filename`。
- 默认最大上传：200 MB（可在 `app.py` 调整）。
- Strategy B/C 当前不处理视频后处理；如需支持，可参考 canvas 渲染逻辑扩展。

## 部署提示

- 生产环境请关闭 `debug=True`，并置于 WSGI / 反向代理之后。
- 确保 `static/uploads/` 与 `static/result/` 可写。
- 前端推理依赖 `onnxruntime-web`（CDN），可按需切换 WASM / WebGL 执行提供器。
