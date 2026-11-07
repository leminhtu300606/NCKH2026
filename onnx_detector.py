"""Backend ONNX FP32 cho detector person (raw head + NMS ngoài).

Không lượng tử hóa. Mục tiêu: thay `YOLO(...)` của Ultralytics trong pipeline
bằng ONNX Runtime raw graph để tránh 2 chi phí:
  1. NMS/TopK nhúng trong graph ONNX (rất chậm trên CPU EP).
  2. Wrapper Ultralytics (letterbox lặp lại, NMS lần 2, copy tensor).

API tương thích tối thiểu với những gì step5 dùng:
    det = OnnxDetector("models/best_raw_dyn.onnx")
    results = det(frame, imgsz=1280, conf=0.0075, iou=0.7)
    boxes = results[0].cpu().boxes
    boxes.xyxy.numpy(), boxes.conf.numpy()
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torchvision.ops import nms
from ultralytics.data.augment import LetterBox

DEFAULT_MODEL = Path("models/best_raw_dyn.onnx")


class OnnxBoxes:
    """Boxes tối giản, tương thích với BYTETracker và step5."""

    def __init__(self, xyxy: torch.Tensor, conf: torch.Tensor, cls: torch.Tensor):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        wh = xyxy[:, 2:4] - xyxy[:, 0:2]
        self.xywh = torch.cat([xyxy[:, 0:2] + wh / 2, wh], dim=1)

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def __getitem__(self, idx: Any) -> "OnnxBoxes":
        return OnnxBoxes(self.xyxy[idx], self.conf[idx], self.cls[idx])

    def cpu(self) -> "OnnxBoxes":
        return self

    def numpy(self) -> "OnnxBoxes":
        return self


class _Result:
    def __init__(self, boxes: OnnxBoxes):
        self.boxes = boxes

    def cpu(self) -> "_Result":
        return self

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> "_Result":
        return self


def _letterbox(
    img: np.ndarray, imgsz: int, stride: int = 32, auto: bool = True
) -> tuple[np.ndarray, float, float, float]:
    """Letterbox y hệt Ultralytics (auto=True => minimum rectangle)."""
    lb = LetterBox((imgsz, imgsz), auto=auto, stride=stride)
    params = lb.get_params({"img": img})
    padded = lb.apply_image({"img": img}, params)["img"]
    r = float(params["ratio"][0])
    pad_w = float(params["left"])
    pad_h = float(params["top"])
    x = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)).astype(np.float32) / 255.0
    return x[None], r, pad_w, pad_h


def decode_boxes(
    out: np.ndarray,
    conf: float,
    iou_thr: float,
    r: float,
    pad_w: float,
    pad_h: float,
    orig_w: int,
    orig_h: int,
    max_det: int = 300,
    max_nms: int = 30000,
) -> "OnnxBoxes":
    a = out[0].T  # (N, 5) = cx, cy, w, h, score
    scores = a[:, 4]
    keep = scores > conf
    a = a[keep]
    scores = scores[keep]
    if len(a) == 0:
        return OnnxBoxes(
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
        )
    if len(a) > max_nms:
        idx = np.argsort(-scores)[:max_nms]
        a, scores = a[idx], scores[idx]
    cx, cy, bw, bh = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / r
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    boxes_t = torch.from_numpy(boxes).float()
    scores_t = torch.from_numpy(scores).float()
    keep_idx = nms(boxes_t, scores_t, iou_thr)[:max_det]
    return OnnxBoxes(
        boxes_t[keep_idx],
        scores_t[keep_idx],
        torch.zeros((len(keep_idx),), dtype=torch.float32),
    )


class OnnxDetector:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        intra_op_threads: int = 8,
        inter_op_threads: int = 1,
        device: str = "auto",
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ONNX model: {model_path}")
        available = ort.get_available_providers()
        if device == "auto":
            device = "cuda:0" if "CUDAExecutionProvider" in available else "cpu"
        self.device = device
        self.use_cuda = device.startswith("cuda") and "CUDAExecutionProvider" in available
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_cuda
            else ["CPUExecutionProvider"]
        )
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if not self.use_cuda:
            so.intra_op_num_threads = intra_op_threads
            so.inter_op_num_threads = inter_op_threads
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.enable_mem_pattern = True
        self.session = ort.InferenceSession(str(model_path), so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.dynamic = any(
            not isinstance(d, int) for d in self.session.get_inputs()[0].shape
        )
        self._buf: np.ndarray | None = None
        self._io = None
        self.use_io_binding = not self.use_cuda

    # -- ONNX Runtime --
    def _run(self, x: np.ndarray) -> np.ndarray:
        if not self.use_io_binding:
            return self.session.run(None, {self.input_name: x})[0]
        if self._io is None or self._buf is None or self._buf.shape != x.shape:
            # Cấp phát buffer cố định và bind một lần cho mỗi shape.
            self._buf = np.empty_like(x)
            self._io = self.session.io_binding()
            self._io.bind_cpu_input(self.input_name, self._buf)
            self._io.bind_output(self.output_name)
        np.copyto(self._buf, x)
        self.session.run_with_iobinding(self._io)
        return self._io.copy_outputs_to_cpu()[0]

    def _decode(
        self,
        out: np.ndarray,
        conf: float,
        iou_thr: float,
        r: float,
        pad_w: float,
        pad_h: float,
        orig_w: int,
        orig_h: int,
        max_det: int = 300,
        max_nms: int = 30000,
    ) -> OnnxBoxes:
        return decode_boxes(
            out, conf, iou_thr, r, pad_w, pad_h, orig_w, orig_h, max_det, max_nms
        )

    # -- API kiểu Ultralytics --
    def __call__(
        self,
        image: np.ndarray,
        imgsz: int = 1280,
        conf: float = 0.0075,
        iou: float = 0.7,
        device: str | None = None,
        verbose: bool = False,
        **_: Any,
    ) -> list[_Result]:
        h, w = image.shape[:2]
        x, r, pad_w, pad_h = _letterbox(image, imgsz, auto=self.dynamic)
        out = self._run(x)
        boxes = self._decode(out, conf, iou, r, pad_w, pad_h, w, h)
        return [_Result(boxes)]


def _openvino_device() -> str:
    """Thiết bị OpenVINO tốt nhất (GPU Intel nếu có), fallback CPU."""
    forced = os.getenv("NCKH_OV_DEVICE")
    if forced:
        return forced
    try:
        from openvino_detector import pick_device

        return pick_device()
    except Exception:
        return "CPU"


def resolve_backend(backend: str = "auto") -> tuple[str, str]:
    """Chọn backend + thiết bị, ưu tiên GPU và fallback CPU.

    Thứ tự khi backend="auto":
      1. PyTorch CUDA (torch.cuda.is_available())
      2. ONNX Runtime CUDAExecutionProvider
      3. OpenVINO GPU (Intel iGPU/Arc; bỏ qua NVIDIA vì plugin rất chậm)
      4. CPU: PyTorch (hoặc NCKH_CPU_BACKEND)
    """
    b = (backend or "auto").lower()
    aliases = {"torch": "pt", "onnxruntime": "onnx"}
    b = aliases.get(b, b)

    if b == "pt":
        return "pt", ("cuda:0" if _torch_cuda() else "cpu")
    if b == "onnx":
        return "onnx", (
            "cuda:0" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
        )
    if b == "openvino":
        return "openvino", _openvino_device()
    if b != "auto":
        raise ValueError(f"Backend không hợp lệ: {backend}")

    if _torch_cuda():
        return "pt", "cuda:0"
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return "onnx", "cuda:0"
    ov_device = _openvino_device()
    if ov_device.upper() != "CPU":
        return "openvino", ov_device
    cpu_backend = os.getenv("NCKH_CPU_BACKEND", "pt").lower()
    if shutil.which("nvidia-smi"):
        print(
            "[Detector] Có NVIDIA GPU nhưng chưa có CUDA runtime khả dụng; "
            "dùng CPU. Cài torch bản CUDA hoặc onnxruntime-gpu để tận dụng GPU."
        )
    return (aliases.get(cpu_backend, cpu_backend), "cpu")


def _torch_cuda() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_detector(
    backend: str = "auto",
    weights: str | Path = "models/best.pt",
    onnx_path: str | Path = DEFAULT_MODEL,
    imgsz: int = 1280,
    static_shape: tuple[int, int] | None = (736, 1280),
):
    """Trả về detector có API tương thích YOLO(...).

    `backend="auto"` ưu tiên GPU, tự fallback CPU nếu không có GPU/runtime.
    """
    kind, device = resolve_backend(backend)
    if kind == "onnx":
        det = OnnxDetector(onnx_path, device=device)
        print(
            f"[Detector] ONNX Runtime [{det.device}] {onnx_path} "
            f"(raw head + NMS ngoài, FP32)"
        )
        return det
    if kind == "openvino":
        from openvino_detector import OpenVinoDetector

        try:
            det = OpenVinoDetector(onnx_path, device=device, static_shape=static_shape)
        except Exception as exc:  # noqa: BLE001
            if device.upper() == "CPU":
                raise
            print(f"[Detector] OpenVINO {device} lỗi: {exc}")
            print("[Detector] Fallback OpenVINO CPU")
            det = OpenVinoDetector(onnx_path, device="CPU", static_shape=static_shape)
        print(
            f"[Detector] OpenVINO FP32 [{det.device}] {onnx_path} "
            f"(static {static_shape}, raw head + NMS ngoài)"
        )
        return det
    if kind not in {"pt", "torch"}:
        raise ValueError(f"Backend không hợp lệ: {backend}")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if device.startswith("cuda"):
        model.to(device)
    print(f"[Detector] PyTorch [{device}] {weights}")
    return model
