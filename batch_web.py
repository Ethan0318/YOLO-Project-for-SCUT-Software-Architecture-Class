# 批量通过 Web 页面测试 A/B/C 策略并收集性能指标（Playwright 驱动）
# 需要本地安装 Microsoft Edge 或 Chromium 等浏览器
import argparse
import csv
import time
import base64
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright

ALLOWED = {"jpg","jpeg","png"}

METRIC_IDS = {
    "a": {"size": "metric-size-a","clientPreUpload": "metric-client-preupload-a","serverRecvPre": "metric-server-recvpre-a","serverInfer": "metric-infer-a","serverPost": "metric-post-a","clientRender": "metric-client-render-a","total": "metric-total-a"},
    "b": {"size": "metric-size-b","clientPreUpload": "metric-client-preupload-b","serverRecvPre": "metric-server-recvpre-b","serverInfer": "metric-infer-b","serverPost": "metric-post-b","clientRender": "metric-client-render-b","total": "metric-total-b"},
    "c": {"size": "metric-size-c","clientPreUpload": "metric-client-preupload-c","serverRecvPre": "metric-server-recvpre-c","serverInfer": "metric-infer-c","serverPost": "metric-post-c","clientRender": "metric-client-render-c","total": "metric-total-c"},
}

def list_images(root: Path):
    """遍历目录树，筛选 `ALLOWED` 扩展名的图片文件"""
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower().lstrip(".") in ALLOWED:
            yield p

def text(page, sel):
    """读取元素文本并去除空白，失败时返回空字符串"""
    try:
        t = page.locator(sel).text_content()
        return (t or "").strip()
    except:
        return ""

def collect_metrics(page, key):
    ids = METRIC_IDS[key]
    return {
        "size": text(page, f'#{ids["size"]}'),
        "clientPreUpload": text(page, f'#{ids["clientPreUpload"]}'),
        "serverRecvPre": text(page, f'#{ids["serverRecvPre"]}'),
        "serverInfer": text(page, f'#{ids["serverInfer"]}'),
        "serverPost": text(page, f'#{ids["serverPost"]}'),
        "clientRender": text(page, f'#{ids["clientRender"]}'),
        "total": text(page, f'#{ids["total"]}'),
    }

def wait_done(page, key, timeout_s):
    ids = METRIC_IDS[key]
    total_id = f'#{ids["total"]}'
    page.wait_for_function(
        "(sel) => { const el = document.querySelector(sel); if (!el) return false; const t = (el.textContent||'').trim(); return t && t !== '—' && t !== '#'; }",
        arg=total_id,
        timeout=timeout_s*1000,
    )

def save_result(page, key, server, out_path: Path):
    """
    保存检测结果图片：优先使用下载链接；否则回退到预览图 `src`。
    支持 `data:` 内联图片与相对路径转绝对 URL。
    """
    href = page.locator(f'#download-{key}-result').get_attribute("href") or ""
    if not href:
        href = page.locator(f'#detected-{key}-image').get_attribute("src") or ""
    if href.startswith("data:"):
        b64 = href.split(",",1)[1]
        data = base64.b64decode(b64)
        out_path.write_bytes(data)
    else:
        # 使用图片实际 src（可能携带缓存参数），相对路径时拼接服务端地址
        if href.startswith("http://") or href.startswith("https://"):
            url = href
        else:
            url = server.rstrip("/") + "/" + href.lstrip("/")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        out_path.write_bytes(r.content)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--strategy", choices=["A","B","C"], required=True)
    ap.add_argument("--server", default="http://localhost:5000")
    ap.add_argument("--output", default=str(Path("d:/Projects/YOLO-Project/batch_test/result")))
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_root = Path(args.output) / args.strategy
    out_root.mkdir(parents=True, exist_ok=True)

    key = args.strategy.lower()

    def parse_mb(text):
        t = (text or "").strip()
        if not t or t in ("—","N/A","#"):
            return None
        try:
            if "MB" in t:
                t = t.replace("MB", "").strip()
            return float(t.split()[0])
        except Exception:
            return None

    def parse_sec(text):
        t = (text or "").strip()
        if not t or t in ("—","N/A","#"):
            return None
        t = t.replace("s", "").strip()
        try:
            return float(t)
        except Exception:
            return None

    def mean(lst):
        return (sum(lst) / len(lst)) if lst else None

    def fmt_mb(v):
        return ("N/A" if v is None else f"{v:.2f} MB")

    def fmt_sec(v):
        return ("N/A" if v is None else f"{v:.3f} s")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=args.headless)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(args.server, timeout=args.timeout*1000)

        csv_path = out_root / "metrics.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["文件名","策略","上传图片大小","客户端预处理与上传","服务器接收与预处理","服务器模型推理","服务器后处理","客户端渲染","端到端耗时","输出文件名"])
            avg = {"size": [], "clientPreUpload": [], "serverRecvPre": [], "serverInfer": [], "serverPost": [], "clientRender": [], "total": []}
            for img_path in list_images(in_dir):
                page.set_input_files("#fileInput", str(img_path))
                page.select_option("#strategySelect", args.strategy)
                page.click("#startBtn")
                wait_done(page, key, args.timeout)
                metrics = collect_metrics(page, key)
                s_mb = parse_mb(metrics["size"])
                if s_mb is not None:
                    avg["size"].append(s_mb)
                cp = parse_sec(metrics["clientPreUpload"])
                if cp is not None:
                    avg["clientPreUpload"].append(cp)
                sr = parse_sec(metrics["serverRecvPre"])
                if sr is not None:
                    avg["serverRecvPre"].append(sr)
                si = parse_sec(metrics["serverInfer"])
                if si is not None:
                    avg["serverInfer"].append(si)
                sp = parse_sec(metrics["serverPost"])
                if sp is not None:
                    avg["serverPost"].append(sp)
                cr = parse_sec(metrics["clientRender"])
                if cr is not None:
                    avg["clientRender"].append(cr)
                tt = parse_sec(metrics["total"])
                if tt is not None:
                    avg["total"].append(tt)
                out_name = f"{img_path.stem}_det.jpg"
                out_path = out_root / out_name
                save_result(page, key, args.server, out_path)
                w.writerow([img_path.name, args.strategy, metrics["size"], metrics["clientPreUpload"], metrics["serverRecvPre"], metrics["serverInfer"], metrics["serverPost"], metrics["clientRender"], metrics["total"], out_name])
            def filt(vals):
                return vals[1:] if args.strategy == "C" and len(vals) > 1 else vals
            m_size = mean(filt(avg["size"])); m_cp = mean(filt(avg["clientPreUpload"])); m_sr = mean(filt(avg["serverRecvPre"])); m_si = mean(filt(avg["serverInfer"])); m_sp = mean(filt(avg["serverPost"])); m_cr = mean(filt(avg["clientRender"])); m_tt = mean(filt(avg["total"]))
            w.writerow(["平均值", args.strategy, fmt_mb(m_size), fmt_sec(m_cp), fmt_sec(m_sr), fmt_sec(m_si), fmt_sec(m_sp), fmt_sec(m_cr), fmt_sec(m_tt), "—"])

        browser.close()

if __name__ == "__main__":
    main()