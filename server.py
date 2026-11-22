#!/usr/bin/env python3
"""Stream images from the original gauss-splat Ascend rasterizer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
from pathlib import Path
import threading
import time

from tunable import options as render_options, render as render_tunable

HERE = Path(__file__).resolve().parent
TILE = 32
COMPARISON_ORDER = ("disabled", "enabled")
RESOLUTIONS = ("192x128", "256x160", "320x192", "640x360", "960x540", "1280x720", "1920x1080")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    **{f"/vendor/{name}": (f"vendor/{name}", "text/javascript; charset=utf-8")
       for name in ("OrbitControls.js", "three.core.min.js", "three.module.min.js")},
}


class FrameError(ValueError):
    pass


def vector(values, count, label, limit=1e6):
    if not isinstance(values, (tuple, list)) or len(values) != count:
        raise FrameError(f"{label} 需要 {count} 个数值")
    if any(type(x) not in (float, int) or abs(x) > limit or not math.isfinite(x) for x in values):
        raise FrameError(f"{label} 包含无效数值")
    return values


def frame_request(data, max_sh):
    if not isinstance(data, dict):
        raise FrameError("请求正文必须是 JSON 对象")

    def integer(key, low, high):
        value = data.get(key)
        if type(value) is not int or not low <= value <= high:
            raise FrameError(f"{key} 必须为 {low}–{high} 内的整数")
        return value

    render_options(data)
    width = integer("width", 64, 3840)
    height = integer("height", 64, 2160)
    degree = integer("sh", 0, max_sh)
    quality = integer("quality", 50, 95)
    fov, = vector([data.get("fov")], 1, "视场角", 90)
    if not 15 <= fov <= 90:
        raise FrameError("视场角必须为 15–90°")
    matrix = vector(data.get("c2w"), 16, "相机矩阵")
    if any(abs(matrix[index] - target) > 1e-5 for index, target in zip(range(12, 16), (0, 0, 0, 1))):
        raise FrameError("相机矩阵末行无效")
    rows = [matrix[i:i+3] for i in (0, 4, 8)]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    if any(abs(dot(a, b) - (i == j)) > 1e-3 for i, a in enumerate(rows) for j, b in enumerate(rows)):
        raise FrameError("相机旋转矩阵必须正交")
    a, b, c = rows
    cross = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    if dot(cross, c) < 0.999:
        raise FrameError("相机旋转矩阵必须是右手坐标系")
    return width, height, degree, quality, fov, matrix


def mask_bytes(width, height, count):
    tiles = math.ceil(width / TILE) * math.ceil(height / TILE)
    return (8 * tiles + 80 * 64) * count


class AscendRenderer:
    def __init__(self, args):
        import torch
        import torch_npu  # Registers the NPU device.
        import gauss_splat
        from PIL import Image

        if not torch.npu.is_available():
            raise RuntimeError("未找到可用 Ascend NPU。请检查 npu-smi、驱动及 CANN 环境")
        from gauss_splat.ops.spherical_harmonics import spherical_harmonics
        self.evaluate_sh = spherical_harmonics
        self.torch, self.image = torch, Image
        self.device = torch.device("npu:0")
        torch.npu.set_device(0)
        self.budget = args.mask_budget_mib * 1024**2

        comparison = None
        if args.comparison_dir:
            comparison = json.loads((args.comparison_dir / "metrics.json").read_text())
            checkpoints = {
                mode: args.comparison_dir / mode / "model.pt"
                for mode in COMPARISON_ORDER
            }
        else:
            checkpoints = {}
        if args.checkpoint:
            checkpoints["default"] = args.checkpoint
        self.models = {mode: self.load_splats(path) for mode, path in checkpoints.items()}
        self.rasterizers = {mode: gauss_splat.Rasterizer() for mode in self.models}
        if comparison:
            for mode in COMPARISON_ORDER:
                if comparison["models"][mode]["count"] != self.models[mode][1]:
                    raise ValueError(f"{mode} 的高斯数量与对照记录不一致")
        _, n, _, box = self.models["enabled" if comparison else "default"]
        degree = min(model[2] for model in self.models.values())
        camera = comparison["camera"] if comparison else {}
        target = args.target or camera.get("target") or [(a+b)/2 for a,b in zip(box[0].tolist(), box[1].tolist())]
        radius = max(0.1, math.dist(*box.tolist()) * .5)
        position = args.position or camera.get("position") or [
            point + radius * shift for point, shift in zip(target, (2.25, -1.65, 1.45))
        ]
        up = args.up or camera.get("up") or [0, 0, 1]
        # A known camera may be much closer than the extent of an outdoor scene.
        if comparison or args.position is not None or args.target is not None:
            radius = max(0.01, math.dist(position, target) / 2.7)
        self.info = {
            "backend": "Ascend NPU", "device": torch.npu.get_device_name(0), "count": n, "maxSh": degree,
            "checkpoint": "Bicycle" if comparison else args.checkpoint.name,
            "target": target, "position": position, "up": up,
            "radius": radius, "maskBudgetMiB": args.mask_budget_mib,
            "initialResolution": args.initial_resolution,
        }
        if comparison:
            self.info["comparison"] = {
                key: comparison[key] for key in ("steps", "testViews", "models")
            }
            mode_counts = {mode: model[1] for mode, model in self.models.items()}
            mode_counts["compare"] = max(mode_counts[mode] for mode in COMPARISON_ORDER)
            self.info["viewModes"] = {
                mode: [
                    size for size in RESOLUTIONS
                    if mask_bytes(*map(int, size.split("x")), count) <= self.budget
                ]
                for mode, count in mode_counts.items()
            }
            if args.checkpoint:
                self.info["originalModel"] = {"count": mode_counts["default"]}
        self.events = None
        try:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            end.record()
            torch.npu.synchronize()
            start.elapsed_time(end)
            self.events = start, end
        except (RuntimeError, AttributeError, TypeError, NotImplementedError):
            print("NPU Event 计时不可用，仍显示同步推理耗时", flush=True)

    def load_splats(self, checkpoint):
        torch = self.torch
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        splats = data.get("splats") if isinstance(data, dict) else None
        if not isinstance(splats, dict) or any(key not in splats for key in ("means", "quats", "scales", "opacities", "sh")):
            raise ValueError("需要 gauss-splat 的 {step, splats} checkpoint")
        means, sh = splats["means"], splats["sh"]
        if means.ndim != 2 or sh.ndim != 3:
            raise ValueError("需要 N-first 形状的 checkpoint")
        n, bases = means.shape[0], sh.shape[1]
        degree = math.isqrt(bases) - 1
        if not n or not 0 <= degree <= 3 or (degree+1)**2 != bases:
            raise ValueError("需要至少一个高斯，且球谐阶数为 0–3")
        expected = {"means": (n, 3), "quats": (n, 4), "scales": (n, 3), "opacities": (n,), "sh": (n, bases, 3)}
        for key, shape in expected.items():
            value = splats[key]
            if not isinstance(value, torch.Tensor) or value.shape != shape or not value.is_floating_point():
                raise ValueError(f"{key} 应为形状 {shape} 的浮点张量")
            splats[key] = value.float()
            if not torch.isfinite(splats[key]).all():
                raise ValueError(f"{key} 包含非有限值")
        parameters = {key: splats[key].T if key in ("means", "quats", "scales") else splats[key] for key in expected}
        parameters["sh"] = splats["sh"].permute(1, 2, 0)
        resident = {key: value.contiguous().to(self.device) for key, value in parameters.items()}

        centers = splats["means"][::max(1, math.ceil(n/65536))]
        box = torch.quantile(centers, torch.tensor([.02, .98]), dim=0)
        return resident, n, degree, box

    def render(self, data):
        torch = self.torch
        width, height, degree, quality, fov, matrix = frame_request(data, self.info["maxSh"])
        renderer_mode, settings = render_options(data)
        mode = data.get("mode", "default" if "default" in self.models else "compare")
        if not isinstance(mode, str):
            raise FrameError("查看模式无效")
        selected = COMPARISON_ORDER if mode == "compare" else (mode,)
        if any(key not in self.models for key in selected):
            raise FrameError("查看模式未配置")
        needed = mask_bytes(width, height, max(self.models[key][1] for key in selected))
        if renderer_mode == "native" and needed > self.budget:
            raise FrameError(f"该分辨率的相交缓冲估算为 {needed / 1024**2:.0f} MiB，超过配置预算，请降低分辨率")
        mem_get_info = getattr(torch.npu, "mem_get_info", None)
        if renderer_mode == "native" and mem_get_info:
            try:
                free, _ = mem_get_info()
            except (RuntimeError, TypeError, NotImplementedError):
                pass
            else:
                cached = max(0, torch.npu.memory_reserved() - torch.npu.memory_allocated())
                if needed > (free + cached) * .8:
                    raise FrameError("NPU 可用内存不足，请降低分辨率或换用较小场景")

        focal = (height * .5) / math.tan(math.radians(fov) * .5)
        inference_ms = event_ms = prepare_ms = 0.0
        frames = []
        with torch.inference_mode():
            camera = torch.tensor(matrix, dtype=torch.float32, device=self.device).reshape(1, 4, 4)
            ks = torch.tensor([[[focal, 0, width*.5], [0, focal, height*.5], [0, 0, 1]]],
                              dtype=torch.float32, device=self.device)
            for key in selected:
                splats = self.models[key][0]
                torch.npu.synchronize()
                start = time.perf_counter()
                if self.events:
                    self.events[0].record()
                if renderer_mode == "tunable":
                    rgb = render_tunable(splats, camera[0], width, height, fov,
                                         degree, settings, self.evaluate_sh)
                else:
                    rgb, _, _ = self.rasterizers[key].rasterization(
                        cam=(camera, ks, "RGB"), size=(width, height), tile_size=TILE,
                        active_sh_degree=degree, splats=splats,
                    )
                if self.events:
                    self.events[1].record()
                torch.npu.synchronize()
                inference_ms += (time.perf_counter() - start) * 1000
                if self.events:
                    event_ms += self.events[0].elapsed_time(self.events[1])

                start = time.perf_counter()
                pixels = rgb[0].float().cpu()
                if pixels.shape != (height, width, 3) or not torch.isfinite(pixels).all():
                    raise RuntimeError("算子输出尺寸错误或包含非有限值")
                pixels = pixels.clamp_(0, 1).mul_(255).round_().to(torch.uint8).numpy()
                frames.append(self.image.fromarray(pixels))
                prepare_ms += (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            combined = frames[0]
            if len(frames) > 1:
                combined = self.image.new("RGB", (width * len(frames), height))
                for index, frame in enumerate(frames):
                    combined.paste(frame, (index * width, 0))
            output = BytesIO()
            combined.save(output, format="JPEG", quality=quality)
            prepare_ms += (time.perf_counter() - start) * 1000
        times = {"X-Inference-Ms": inference_ms, "X-Prepare-Ms": prepare_ms}
        if self.events:
            times["X-Npu-Ms"] = event_ms
        return output.getvalue(), times


class ViewerHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, info, submit, **kwargs):
        self.info, self.submit = info, submit
        super().__init__(*args, **kwargs)

    def send_bytes(self, code, body, kind, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, f"{value:.3f}")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, code, data):
        self.send_bytes(code, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path == "/api/info":
            self.send_json(200, self.info)
        elif self.path in ASSETS:
            name, content_type = ASSETS[self.path]
            self.send_bytes(200, (HERE / "static" / name).read_bytes(), content_type)
        elif self.path == "/favicon.ico":
            self.send_bytes(204, b"", "text/plain")
        else:
            self.send_json(404, {"error": "路径不存在"})

    def do_POST(self):
        if self.path != "/api/frame":
            self.send_json(404, {"error": "路径不存在"})
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(415, {"error": "只接受 application/json"})
            return
        try:
            count = int(self.headers.get("Content-Length", "0"))
            if not 1 <= count <= 4096:
                raise FrameError("请求长度超限")
            data = json.loads(self.rfile.read(count))
            frame_request(data, self.info["maxSh"])
            result = self.submit(data)
        except (ValueError, UnicodeError) as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            print(f"Render error: {type(error).__name__}: {error}", flush=True)
            self.send_json(500, {"error": "渲染失败，请检查服务端日志"})
        else:
            if result is None:
                self.send_json(429, {"error": "服务器当前正在渲染另一帧"})
            else:
                body, timings = result
                self.send_bytes(200, body, "image/jpeg", timings)

    def log_request(self, code="-", size="-"):
        if code != 200 or self.path != "/api/frame":
            super().log_request(code, size)


def smoke(renderer, output, renderer_mode="native"):
    from PIL import Image

    info = renderer.info
    eye, target, up = (renderer.torch.tensor(info[key], dtype=renderer.torch.float32)
                       for key in ("position", "target", "up"))
    forward = renderer.torch.nn.functional.normalize(target-eye, dim=0)
    right = renderer.torch.nn.functional.normalize(renderer.torch.linalg.cross(forward, up), dim=0)
    down = renderer.torch.linalg.cross(forward, right)
    camera = renderer.torch.eye(4)
    camera[:3] = renderer.torch.stack((right, down, forward, eye), dim=1)
    comparing = "comparison" in info
    width, height = (320, 240)
    if comparing:
        options = info["viewModes"]["compare"]
        if not options:
            raise FrameError("当前内存预算无法加载稠密化对照，请检查目标卡可用内存和 --mask-budget-mib")
        size = info["initialResolution"] if info["initialResolution"] in options else options[0]
        width, height = map(int, size.split("x"))
    body, times = renderer.render(dict(width=width, height=height, sh=info["maxSh"], quality=88,
                                       fov=39, c2w=camera.flatten().tolist(),
                                       mode="compare" if comparing else "default", renderer=renderer_mode))
    with Image.open(BytesIO(body)) as frame:
        sides = len(COMPARISON_ORDER) if comparing else 1
        if frame.size != (width * sides, height):
            raise RuntimeError("自检画面的尺寸错误")
        for offset in range(0, frame.width, width):
            pane = frame.crop((offset, 0, offset + width, height))
            if max(channel[1] for channel in pane.getextrema()) <= 2:
                raise RuntimeError("自检相机画面为空，请核对模型坐标、--position 和 --target")
    if output:
        output.write_bytes(body)
    print(json.dumps({**info, **times}, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--comparison-dir", type=Path, help="Bicycle 对照模型与指标目录")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--mask-budget-mib", type=int, default=1024)
    parser.add_argument("--initial-resolution", choices=RESOLUTIONS, default="640x360")
    parser.add_argument("--position", nargs=3, type=float)
    parser.add_argument("--target", nargs=3, type=float)
    parser.add_argument("--up", nargs=3, type=float)
    parser.add_argument("--check", action="store_true", help="render a test frame and exit")
    parser.add_argument("--check-renderer", choices=("native", "tunable"), default="native",
                        help="renderer exercised by --check")
    parser.add_argument("--output", type=Path, help="write the JPEG produced by --check")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.mask_budget_mib <= 0:
        parser.error("检查 checkpoint 路径、端口和显存预算")
    if args.comparison_dir:
        required = ["metrics.json", *(f"{mode}/model.pt" for mode in COMPARISON_ORDER)]
        if any(not (args.comparison_dir / name).is_file() for name in required):
            parser.error("对照目录需要 metrics.json 和 enabled、disabled 两组 model.pt")
    elif args.checkpoint is None:
        args.checkpoint = HERE / "lego.pt"
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error("checkpoint 不存在")
    if args.output and not args.check:
        parser.error("--output 需与 --check 同用")
    for name in ("position", "target", "up"):
        if getattr(args, name) is not None:
            vector(getattr(args, name), 3, name)
    with ThreadPoolExecutor(max_workers=1) as worker:
        try:
            renderer = worker.submit(AscendRenderer, args).result()
            if args.check:
                worker.submit(smoke, renderer, args.output, args.check_renderer).result()
                return
        except ImportError as error:
            parser.exit(1, f"无法加载昇腾运行依赖：{error}\n请核对 README 中的 CANN、torch_npu 和 gauss-splat wheel。\n")
        gate = threading.Lock()

        def submit(data):
            if not gate.acquire(blocking=False):
                return None
            try:
                return worker.submit(renderer.render, data).result()
            finally:
                gate.release()

        handler = partial(ViewerHandler, info=renderer.info, submit=submit)
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            print(f'{renderer.info["device"]} | {renderer.info["count"]:,} Gaussians | http://127.0.0.1:{args.port}/', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
