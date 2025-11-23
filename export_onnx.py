from ultralytics import YOLO

# 加载你刚刚成功下载的 yolov8n.pt
model = YOLO("yolov8n.pt")

# 导出 ONNX
model.export(format="onnx", opset=12, dynamic=False)

print("ONNX 模型导出成功！")
