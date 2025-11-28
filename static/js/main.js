const fileInput = document.getElementById("fileInput");
const fileInfo = document.getElementById("fileInfo");
const strategySelect = document.getElementById("strategySelect");
const startBtn = document.getElementById("startBtn");
const spinner = document.getElementById("spinner");
const progressBar = document.getElementById("progressBar");
const statusText = document.getElementById("statusText");

const strategyPanels = {
  A: buildPanel("a"),
  B: buildPanel("b"),
  C: buildPanel("c"),
};

const metricIds = {
  a: { upload: "metric-upload-a", server_pre: "metric-serverpre-a", download: "metric-download-a", total: "metric-total-a" },
  b: { client_pre: "metric-clientpre-b", upload: "metric-upload-b", server_pre: "metric-serverpre-b", download: "metric-download-b", total: "metric-total-b" },
  c: { upload: "metric-upload-c", download: "metric-download-c", total: "metric-total-c" },
};

const COCO_CLASSES = [
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
  "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
  "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
  "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
  "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
  "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
  "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
  "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
  "scissors","teddy bear","hair drier","toothbrush"
];

const IMG_SIZE = 640;
const CONF_THRESHOLD_C = 0.25;
const IOU_THRESHOLD_C = 0.45;
const PRE_NMS_TOPK = 600;
const MAX_DETS = 300;
const CLASS_AGNOSTIC_NMS_C = false;

let progressInterval = null;
let ortSession = null;

function buildPanel(key) {
  return {
    originalImage: document.getElementById(`original-${key}-image`),
    detectedImage: document.getElementById(`detected-${key}-image`),
    canvas: document.getElementById(`detected-${key}-canvas`),
    downloadOriginal: document.getElementById(`download-${key}-original`),
    downloadResult: document.getElementById(`download-${key}-result`),
  };
}

fileInput.addEventListener("change", () => {
  if (!fileInput.files.length) {
    fileInfo.textContent = "未选择文件";
    return;
  }
  const file = fileInput.files[0];
  if (file.type.startsWith("video")) {
    alert("不支持视频上传，请选择图片 (jpg/jpeg/png)");
    fileInput.value = "";
    fileInfo.textContent = "未选择文件";
    return;
  }
  fileInfo.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
});

startBtn.addEventListener("click", () => {
  if (!fileInput.files.length) {
    alert("请先选择文件");
    return;
  }
  const file = fileInput.files[0];
  const strategy = strategySelect.value.toUpperCase();

  if (file.type.startsWith("video")) {
    setStatus("当前不支持视频上传，请选择图片。");
    alert("当前不支持视频上传，请选择图片 (jpg/jpeg/png)。");
    return;
  }

  resetStrategyMedia(strategy);
  resetMetrics(strategy);
  setStatus("开始处理...");
  startLoading();



  showOriginalPreview(file, strategy);

  if (strategy === "C") {
    handleStrategyC(file).catch(handleError);
  } else {
    handleServerStrategy(file, strategy).catch(handleError);
  }
});

function handleError(err) {
  console.error(err);
  setStatus(err.message || "发生错误");
  alert(err.message || "请求失败");
  stopLoading();
}

async function handleServerStrategy(file, strategy) {
  const overallStart = performance.now();

  const formData = new FormData();
  let mapping = null;
  let clientPreTime = 0;

  if (strategy === "B") {
    const clientPreStart = performance.now();
    const img = await loadImage(URL.createObjectURL(file));
    const { blob, ratio, padX, padY } = await letterboxToBlob(img, IMG_SIZE);
    const smallFile = new File([blob], `resized_${Date.now()}.png`, { type: "image/png" });

    formData.append("file", smallFile);
    formData.append("strategy", strategy);
    formData.append("orig_w", String(img.naturalWidth || img.width));
    formData.append("orig_h", String(img.naturalHeight || img.height));
    formData.append("ratio", String(ratio));
    formData.append("pad_x", String(padX));
    formData.append("pad_y", String(padY));
    clientPreTime = (performance.now() - clientPreStart) / 1000;

    mapping = { ratio, padX, padY, origW: img.naturalWidth || img.width, origH: img.naturalHeight || img.height };
  } else {
    formData.append("file", file);
    formData.append("strategy", strategy);
  }

  const resp = await fetch("/detect", { method: "POST", body: formData });

  let rawText = "";
  let responseDownTTLB = 0;
  if (resp.body && resp.body.getReader) {
    const reader = resp.body.getReader();
    const chunks = [];
    const decoder = new TextDecoder("utf-8");
    const downStart = performance.now();
    while (true) {
      const { done, value } = await reader.read();
      if (value) chunks.push(value);
      if (done) break;
    }
    responseDownTTLB = (performance.now() - downStart) / 1000;
    const totalLen = chunks.reduce((n, c) => n + c.length, 0);
    const merged = new Uint8Array(totalLen);
    let offset = 0;
    for (const c of chunks) { merged.set(c, offset); offset += c.length; }
    rawText = decoder.decode(merged);
  } else {
    const downStart = performance.now();
    rawText = await resp.text();
    responseDownTTLB = (performance.now() - downStart) / 1000;
  }

  let data;
  try {
    data = JSON.parse(rawText);
  } catch (_) {
    data = {};
  }
  if (!resp.ok) {
    let msg = "服务器错误";
    try {
      msg = (data && data.error) || msg;
    } catch (_) {}
    throw new Error(msg);
  }

  if (strategy === "A") {
    await handleStrategyAResult(data, file, overallStart, responseDownTTLB);
  } else {
    await handleStrategyBResult(data, file, overallStart, responseDownTTLB, mapping, clientPreTime);
  }
  stopLoading();
}

async function handleStrategyAResult(data, file, overallStart, responseDownTTLB) {
  const panel = strategyPanels.A;
  const localUrl = panel.originalImage?.src || URL.createObjectURL(file);
  panel.downloadOriginal.href = localUrl;
  showOriginalMedia(panel, localUrl);

  const resultUrl = `/${data.detected}?t=${Date.now()}`;
  hideElements(panel.canvas);
  panel.detectedImage.classList.remove("hidden");
  await waitImageLoad(panel.detectedImage, resultUrl);
  panel.downloadResult.href = `/${data.detected}`;

  const overall = (performance.now() - overallStart) / 1000;
  setMetricCell("a", "upload", formatSeconds(data.upload_time ?? "N/A"));
  setMetricCell("a", "server_pre", formatSeconds(data.server_pre_time ?? "N/A"));
  setMetricCell("a", "download", formatSeconds(responseDownTTLB));
  setMetricCell("a", "total", formatSeconds(overall));
  setStatus("Strategy A 完成");
}

async function handleStrategyBResult(data, file, overallStart, responseDownTTLB, mapping, clientPreTime) {
  const panel = strategyPanels.B;
  const localUrl = panel.originalImage?.src || URL.createObjectURL(file);
  panel.downloadOriginal.href = localUrl;

  const boxesSmall = data.boxes || [];
  const boxes = Array.isArray(boxesSmall) && mapping
    ? boxesSmall.map((b) => {
        const rect = deLetterBox(b.x1, b.y1, b.x2, b.y2, mapping.ratio, mapping.padX, mapping.padY, mapping.origW, mapping.origH);
        return { ...rect, cls: b.cls, conf: b.conf };
      })
    : boxesSmall;

  const { dataUrl, clientTime } = await drawBoundingBoxes(panel.originalImage, boxes, panel.canvas, { color: "#0ea5e9" });
  const overall = (performance.now() - overallStart) / 1000;

  showDetectedCanvas(panel);
  panel.detectedImage.src = dataUrl;
  panel.canvas.classList.remove("hidden");
  panel.downloadResult.href = dataUrl;

  setMetricCell("b", "client_pre", formatSeconds(clientPreTime));
  setMetricCell("b", "upload", formatSeconds(data.upload_time ?? "N/A"));
  setMetricCell("b", "server_pre", formatSeconds("N/A"));
  setMetricCell("b", "download", formatSeconds(responseDownTTLB));
  setMetricCell("b", "total", formatSeconds(overall));
  setStatus("Strategy B 完成");
}

async function handleStrategyC(file) {
  const panel = strategyPanels.C;
  const overallStart = performance.now();
  const img = await loadImage(URL.createObjectURL(file));
  const { tensor, ratio, padX, padY } = preprocessImage(img, IMG_SIZE);

  if (!ortSession) {
    setStatus("加载浏览器模型...");
    ortSession = await ort.InferenceSession.create("/static/js/yolov8n.onnx", {
      executionProviders: ["wasm", "webgl"],
    });
  }

  setStatus("浏览器推理中...");
  const inferStart = performance.now();
  const outputs = await ortSession.run({ [ortSession.inputNames[0]]: tensor });
  const inferTime = (performance.now() - inferStart) / 1000;

  const outputTensor = outputs[ortSession.outputNames[0]];
  const clientStart = performance.now();
  const decoded = decodeDetections(outputTensor, ratio, padX, padY, img.naturalWidth, img.naturalHeight, CONF_THRESHOLD_C, PRE_NMS_TOPK);
  const finalBoxes = nonMaxSuppression(decoded, IOU_THRESHOLD_C, MAX_DETS);

  setStatus("客户端后处理...");
  const { dataUrl, clientTime } = await drawBoundingBoxes(img, finalBoxes, panel.canvas, { color: "#f97316" });
  const clientProcTime = (performance.now() - clientStart) / 1000 + clientTime;
  const overall = (performance.now() - overallStart) / 1000;

  showOriginalMedia(panel, img.src, false);
  showDetectedCanvas(panel);
  panel.detectedImage.src = dataUrl;
  panel.downloadOriginal.href = img.src;
  panel.downloadResult.href = dataUrl;

  setMetricCell("c", "upload", "N/A");
  setMetricCell("c", "download", "N/A");
  setMetricCell("c", "total", formatSeconds(overall));
  stopLoading();
  setStatus("Strategy C 完成");
  console.log("Client-side infer:", inferTime);
}

function showOriginalPreview(file, strategy) {
  const url = URL.createObjectURL(file);
  const panel = strategyPanels[strategy];
  showOriginalMedia(panel, url);
  panel.downloadOriginal.href = url;
}

function showOriginalMedia(panel, url) {
  panel.originalImage.classList.remove("hidden");
  panel.originalImage.src = url;
}

function showDetectedImage(panel, url) {
  hideElements(panel.canvas);
  panel.detectedImage.classList.remove("hidden");
  panel.detectedImage.src = url;
}



function showDetectedCanvas(panel) {
  hideElements(panel.detectedImage);
  if (panel.canvas) {
    panel.canvas.classList.remove("hidden");
  }
}

async function drawBoundingBoxes(imageSource, boxes, targetCanvas, options = {}) {
  const canvas = targetCanvas;
  const ctx = canvas.getContext("2d");

  const img = await loadImage(imageSource);

  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const procStart = performance.now();
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  if (boxes.length) {
    boxes.forEach((b) => {
      const color = options.color || "#0ea5e9";
      const base = Math.min(canvas.width, canvas.height);
      const lw = options.lineWidth ?? Math.max(6, Math.round(base / 120));
      const fontSize = options.fontSize ?? Math.max(22, Math.round(lw * 7));
      const padding = Math.max(6, Math.round(lw * 2));

      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

      const label = `${COCO_CLASSES[b.cls] || "obj"} ${b.conf.toFixed(2)}`;
      ctx.font = `${fontSize}px Manrope, sans-serif`;
      const textWidth = ctx.measureText(label).width + padding * 2;
      const textHeight = fontSize + padding;
      ctx.fillStyle = color;
      const rectY = Math.max(0, b.y1 - textHeight);
      ctx.fillRect(b.x1, rectY, textWidth, textHeight);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, b.x1 + padding, rectY + fontSize);
    });
  }

  const clientTime = (performance.now() - procStart) / 1000;
  const dataUrl = canvas.toDataURL("image/png");
  return { dataUrl, clientTime };
}

function preprocessImage(img, size) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = size;
  canvas.height = size;

  const ratio = Math.min(size / img.naturalWidth, size / img.naturalHeight);
  const newW = Math.round(img.naturalWidth * ratio);
  const newH = Math.round(img.naturalHeight * ratio);
  const padX = (size - newW) / 2;
  const padY = (size - newH) / 2;

  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(img, padX, padY, newW, newH);

  const imageData = ctx.getImageData(0, 0, size, size).data;
  const floatData = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    floatData[i] = imageData[i * 4] / 255;
    floatData[i + size * size] = imageData[i * 4 + 1] / 255;
    floatData[i + 2 * size * size] = imageData[i * 4 + 2] / 255;
  }
  const tensor = new ort.Tensor("float32", floatData, [1, 3, size, size]);
  return { tensor, ratio, padX, padY };
}

async function letterboxToBlob(img, size) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = size;
  canvas.height = size;

  const srcW = img.naturalWidth || img.width;
  const srcH = img.naturalHeight || img.height;
  const ratio = Math.min(size / srcW, size / srcH);
  const newW = Math.round(srcW * ratio);
  const newH = Math.round(srcH * ratio);
  const padX = (size - newW) / 2;
  const padY = (size - newH) / 2;

  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(img, padX, padY, newW, newH);

  const blob = await new Promise((resolve) => canvas.toBlob((b) => resolve(b), "image/png"));
  return { blob, ratio, padX, padY };
}

function decodeDetections(outputTensor, ratio, padX, padY, origW, origH, confThres = CONF_THRESHOLD_C, topK = PRE_NMS_TOPK) {
  const data = outputTensor.data;
  const dims = outputTensor.dims;
  const numClasses = COCO_CLASSES.length;

  let channels = 0;
  let numAnchors = 0;
  let transposed = false;
  if (dims.length === 3) {
    if (dims[1] === numClasses + 4 || dims[1] === numClasses + 5) {
      channels = dims[1];
      numAnchors = dims[2];
      transposed = true;
    } else if (dims[2] === numClasses + 4 || dims[2] === numClasses + 5) {
      channels = dims[2];
      numAnchors = dims[1];
    } else {
      channels = (data.length % (numClasses + 5) === 0) ? (numClasses + 5) : (numClasses + 4);
      numAnchors = data.length / channels;
    }
  } else {
    channels = (data.length % (numClasses + 5) === 0) ? (numClasses + 5) : (numClasses + 4);
    numAnchors = data.length / channels;
  }
  const hasObj = channels === numClasses + 5;
  const clsOffset = hasObj ? 5 : 4;

  const boxes = [];
  for (let i = 0; i < numAnchors; i++) {
    let x, y, w, h, objProb = 1;
    if (transposed) {
      x = data[i];
      y = data[numAnchors + i];
      w = data[2 * numAnchors + i];
      h = data[3 * numAnchors + i];
      if (hasObj) objProb = toProb(data[4 * numAnchors + i]);
    } else {
      const offset = i * channels;
      x = data[offset];
      y = data[offset + 1];
      w = data[offset + 2];
      h = data[offset + 3];
      if (hasObj) objProb = toProb(data[offset + 4]);
    }

    let bestConf = 0;
    let bestCls = -1;
    for (let c = 0; c < numClasses; c++) {
      const raw = transposed ? data[(clsOffset + c) * numAnchors + i] : data[i * channels + clsOffset + c];
      const clsProb = toProb(raw);
      const conf = hasObj ? clsProb * objProb : clsProb;
      if (conf > bestConf) {
        bestConf = conf;
        bestCls = c;
      }
    }
    if (bestConf < confThres) continue;

    const [x1, y1, x2, y2] = xywhToXyxy(x, y, w, h);
    const rect = deLetterBox(x1, y1, x2, y2, ratio, padX, padY, origW, origH);
    boxes.push({ ...rect, cls: bestCls, conf: Math.max(0, Math.min(1, bestConf)) });
  }
  boxes.sort((a, b) => b.conf - a.conf);
  return boxes.slice(0, topK);
}

function nonMaxSuppression(boxes, iouThreshold, maxDet = MAX_DETS) {
  if (CLASS_AGNOSTIC_NMS_C) {
    const list = boxes.slice().sort((a, b) => b.conf - a.conf);
    const picked = [];
    while (list.length && picked.length < maxDet) {
      const candidate = list.shift();
      picked.push(candidate);
      for (let i = list.length - 1; i >= 0; i--) {
        if (iou(candidate, list[i]) > iouThreshold) {
          list.splice(i, 1);
        }
      }
    }
    return picked;
  }
  const byClass = {};
  boxes.forEach((b) => {
    if (!byClass[b.cls]) byClass[b.cls] = [];
    byClass[b.cls].push(b);
  });
  const picked = [];
  Object.values(byClass).forEach((list) => {
    list.sort((a, b) => b.conf - a.conf);
    while (list.length) {
      const candidate = list.shift();
      picked.push(candidate);
      for (let i = list.length - 1; i >= 0; i--) {
        if (iou(candidate, list[i]) > iouThreshold) {
          list.splice(i, 1);
        }
      }
    }
  });
  picked.sort((a, b) => b.conf - a.conf);
  return picked.slice(0, maxDet);
}

function deLetterBox(x1, y1, x2, y2, ratio, padX, padY, origW, origH) {
  const left = clamp((x1 - padX) / ratio, 0, origW);
  const top = clamp((y1 - padY) / ratio, 0, origH);
  const right = clamp((x2 - padX) / ratio, 0, origW);
  const bottom = clamp((y2 - padY) / ratio, 0, origH);
  return { x1: left, y1: top, x2: right, y2: bottom };
}

function xywhToXyxy(x, y, w, h) {
  const halfW = w / 2;
  const halfH = h / 2;
  return [x - halfW, y - halfH, x + halfW, y + halfH];
}

function iou(a, b) {
  const interArea = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1)) *
    Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const unionArea =
    (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - interArea;
  return unionArea === 0 ? 0 : interArea / unionArea;
}

function loadImage(srcOrImg) {
  return new Promise((resolve, reject) => {
    if (srcOrImg instanceof HTMLImageElement) {
      if (srcOrImg.complete) return resolve(srcOrImg);
      srcOrImg.onload = () => resolve(srcOrImg);
      srcOrImg.onerror = reject;
      return;
    }
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = srcOrImg;
  });
}

function waitImageLoad(imgEl, url) {
  return new Promise((resolve) => {
    const t0 = performance.now();
    imgEl.onload = () => resolve((performance.now() - t0) / 1000);
    imgEl.onerror = () => resolve((performance.now() - t0) / 1000);
    imgEl.src = url;
  });
}



function setMetricCell(strategy, key, value) {
  const id = metricIds[strategy][key];
  if (!id) return;
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function resetMetrics(strategy) {
  const key = strategy.toLowerCase();
  const group = metricIds[key];
  if (!group) return;
  Object.values(group).forEach((id) => {
    if (id) {
      const cell = document.getElementById(id);
      if (cell) cell.textContent = "—";
    }
  });
}

function resetStrategyMedia(strategy) {
  const panel = strategyPanels[strategy];
  if (!panel) return;
  hideElements(panel.originalImage, panel.detectedImage, panel.canvas);
  if (panel.canvas) {
    panel.canvas.getContext("2d").clearRect(0, 0, panel.canvas.width || 0, panel.canvas.height || 0);
    panel.canvas.width = panel.canvas.height = 0;
  }
  panel.downloadOriginal.href = "#";
  panel.downloadResult.href = "#";
}

function hideElements(...els) {
  els.forEach((el) => {
    if (el) el.classList.add("hidden");
  });
}

function startLoading() {
  spinner.classList.remove("hidden");
  statusText.classList.remove("error");
  progressBar.style.width = "8%";
  if (progressInterval) clearInterval(progressInterval);
  let value = 8;
  progressInterval = setInterval(() => {
    value = Math.min(96, value + Math.random() * 8);
    progressBar.style.width = `${value}%`;
  }, 300);
}

function stopLoading() {
  spinner.classList.add("hidden");
  if (progressInterval) {
    clearInterval(progressInterval);
    progressInterval = null;
  }
  progressBar.style.width = "100%";
  setTimeout(() => (progressBar.style.width = "0%"), 500);
}

function setStatus(text) {
  statusText.textContent = text;
}

function formatSeconds(sec) {
  if (sec === "N/A") return "N/A";
  return `${Number(sec).toFixed(3)} s`;
}

function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}
function toProb(v) {
  if (v >= 0 && v <= 1) return v;
  return 1 / (1 + Math.exp(-v));
}
