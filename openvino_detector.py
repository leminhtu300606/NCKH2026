"""Backend OpenVINO (Intel CPU) cho detector person, dùng chung decode với ONNX.

Không lượng tử hóa: giữ FP32. OpenVINO đọc trực tiếp file ONNX raw-head.

API tương thích tối thiểu với step5, giống OnnxDetector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov

from onnx_detector import OnnxBoxes, _Result, _letterbox, decode_boxes


def gpu_devices() -> list[str]:
    """Danh sách GPU mà OpenVINO nhìn thấy (ví dụ GPU.0, GPU.1)."""
    try:
        return [d for d in ov.Core().available_devices if d.startswith("GPU")]
    except Exception:
        return []


def pick_device(core: "ov.Core | None" = None, allow_nvidia: bool | None = None) -> str:
    """Chọn GPU tốt nhất có thể, fallback CPU.

    Mặc định ưu tiên GPU Intel (iGPU/Arc) vì plugin GPU của OpenVINO chạy tốt;
    GPU NVIDIA chỉ dùng khi `NCKH_OV_ALLOW_NVIDIA=1` (đo thực tế thấy rất chậm).
    """
    import os

    if allow_nvidia is None:
        allow_nvidia = os.getenv("NCKH_OV_ALLOW_NVIDIA", "0") == "1"
    core = core or ov.Core()
    gpus = [d for d in core.available_devices if d.startswith("GPU")]
    if not gpus:
        return "CPU"

    names: dict[str, str] = {}
    for dev in gpus:
        try:
            names[dev] = str(core.get_property(dev, "FULL_DEVICE_NAME")).lower()
        except Exception:
            names[dev] = ""

    intel_keys = ("intel", "iris", "arc", " xe")
    for dev in gpus:
        if any(k in names[dev] for k in intel_keys):
            return dev
    if allow_nvidia:
        nvidia_keys = ("nvidia", "geforce", "rtx")
        for dev in gpus:
            if any(k in names[dev] for k in nvidia_keys):
                return dev
    others = [
        d for d in gpus
        if not any(k in names[d] for k in ("nvidia", "geforce", "rtx"))
    ]
    return others[0] if others else "CPU"


class OpenVinoDetector:
    def __init__(
        self,
        model_path: str | Path = "models/best_raw_dyn.onnx",
        device: str = "CPU",
        num_threads: int = 8,
        static_shape: tuple[int, int] | None = None,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
        self.core = ov.Core()
        if device.upper() == "AUTO":
            device = pick_device(self.core)
        self.device = device
        model = self.core.read_model(str(model_path))
        if static_shape is not None:
            h, w = static_shape
            model.reshape({0: [1, 3, h, w]})
        config: dict[str, Any] = {"PERFORMANCE_HINT": "LATENCY"}
        if device.upper() == "CPU":
            config["INFERENCE_NUM_THREADS"] = num_threads
            config["NUM_STREAMS"] = 1
        self.compiled = self.core.compile_model(model, device, config)
        self.request = self.compiled.create_infer_request()
        self.input_key = self.compiled.input(0)
        self.output_key = self.compiled.output(0)
        self.static_shape = static_shape
        self.dynamic = static_shape is None

    def _run(self, x: np.ndarray) -> np.ndarray:
        self.request.infer({self.input_key: x})
        return self.request.get_output_tensor(0).data

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
        x, r, pad_w, pad_h = _letterbox(image, imgsz, auto=True)
        if self.static_shape is not None and x.shape[2:] != tuple(self.static_shape):
            raise ValueError(
                f"Input {x.shape[2:]} không khớp static_shape {self.static_shape}"
            )
        out = self._run(x)
        boxes = decode_boxes(out, conf, iou, r, pad_w, pad_h, w, h)
        return [_Result(boxes)]
