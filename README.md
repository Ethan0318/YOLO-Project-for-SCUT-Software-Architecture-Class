# 基于YOLOv8的图片检测系统

基于 Flask + YOLOv8 的目标检测演示，支持服务器 / 浏览器混合分区推理。三种策略可切换并对比耗时：

- **Strategy A**：纯服务器推理与画框
- **Strategy B**：客户端预处理，服务器推理，客户端画框
- **Strategy C**：纯前端（浏览器 ONNX/Web 推理与画框）

## ver-3.0 更新说明

- 调整了性能检测模块的指标。由于网络传播延迟需要同时采集客户端发送首字节和服务器接收首字节的时间，而二者的时钟不统一，只能进行近似估算，近似估算非常不准确，因此不再采用。取而代之的是，上传图片大小、客户端预处理与上传、服务器接收与预处理、服务器模型推理、服务器后处理、客户端渲染、端到端耗时这七个指标，这七个指标定义明确，易于计算，且能体现策略间的差异。
- 调整了前端界面，新增了一些说明，修改了排版。

## ver-2.0 更新说明

- 前端显示的“原始”预览均改为本地 object URL
- 策略b增加了预处理功能，在前端先缩放到 640×640 后上传小图（前端缩放、参数随表单上传、服务端按小图推理、前端坐标反变换与绘制），使其性能极大提升
- 调整了性能检测模块的指标，改为上传图片大小、网络传播延迟（上传和下载的总延迟）端到端耗时这三个指标，更能体现实验要求

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
   - **A**：仅服务器侧执行；客户端上传原图，服务器完成预处理、推理与标注后返回成图。
   - **B**：前后端协同；客户端做轻量预处理并上传，服务器仅推理返回检测框，客户端本地绘制。
   - **C**：纯前端；浏览器使用 Web ONNX 在本地完成预处理、推理与绘制，无需服务器参与。
3. 点击“开始检测”，查看状态、进度与性能表。
4. 各策略有独立结果面板和下载链接，依次运行 A→B→C 也不会互相覆盖。

## 性能指标

- 上传图片大小：A=原始图片体积；B=客户端预处理后上传的图片体积；C=N/A。
- 客户端预处理与上传：A=本地读取+直传；B=读取+压缩/缩放/归一化后上传；C=N/A。
- 服务器接收与预处理：A=接收原图、解码并构建模型输入；B=接收已预处理图并做轻量转换/校验；C=N/A。
- 服务器模型推理：A/B=服务器前向计算耗时；C=N/A。
- 服务器后处理：A=NMS+绘制+编码成图后返回；B=仅NMS与结果序列化（不绘制）；C=N/A。
- 客户端渲染：A=加载显示服务器返回成图；B=浏览器中绘制检测框/标签并生成结果图；C=N/A。
- 端到端耗时：A/B/C=从点击开始到结果呈现的总耗时（含各自适用阶段）。

## 安全与限制

- 上传白名单：`jpg`、`jpeg`、`png`；使用 `secure_filename`。
- 默认最大上传：200 MB（可在 `app.py` 调整）。

## 部署提示

- 生产环境请关闭 `debug=True`，并置于 WSGI / 反向代理之后。
- 确保 `static/uploads/` 与 `static/result/` 可写。
- 前端推理依赖 `onnxruntime-web`（CDN），可按需切换 WASM / WebGL 执行提供器。
