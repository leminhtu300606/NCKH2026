"""Tiện ích thiết bị và hiệu năng dùng chung cho Bước 5 và Bước 6.

Ba mục tiêu:

1. **Không âm thầm chạy CPU khi máy có GPU NVIDIA.** Nếu `torch.cuda.is_available()`
   là False nhưng `nvidia-smi` tồn tại (máy có GPU), in cảnh báo rõ kèm lệnh sửa.
   Lỗi cài `pip install -r requirements.txt` (bản `+cpu`) là nguyên nhân phổ biến
   nhất khiến pipeline chỉ đạt ~3 FPS trên máy có RTX.
2. **Blur theo ROI** thay vì GaussianBlur toàn frame. Kết quả trong vùng che
   giống hệt blur toàn frame (cùng kernel, crop có margin >= kernel/2) nhưng chỉ
   xử lý diện tích thật sự cần che.
3. **Gom cấu hình thread và FP16** vào một chỗ, có thể override bằng biến môi trường.

Biến môi trường liên quan:

- `NCKH_DEVICE`        : ép thiết bị, ví dụ `cpu`, `cuda:0`, `cuda:1`.
- `NCKH_HALF`          : `1`/`0` bật/tắt FP16. Mặc định bật khi chạy CUDA + PyTorch.
- `NCKH_BLUR_MODE`     : `roi` (mặc định) hoặc `full` (cách cũ, để đối chứng).
- `NCKH_BLUR_KERNEL`   : kích thước kernel Gaussian, mặc định 15 (đồng bộ
                         `ANON_KSIZE` của notebook4234ae1bfe).
- `NCKH_CV_THREADS`    : số thread OpenCV; mặc định `min(4, số CPU)` vì
                         `import ultralytics` đã tắt thread OpenCV
                         (`cv2.setNumThreads(0)`), làm blur/SSIM chậm 1,5-2x.
- `NCKH_TORCH_THREADS` : số thread PyTorch; không đặt thì chỉ giới hạn khi chạy CUDA.
- `NCKH_DECODE_THREADS`: số thread đọc JPEG trước (0 = đọc tuần tự).
- `NCKH_ENCODE_QUEUE`  : số frame xếp hàng cho thread ghi video (0 = ghi đồng bộ).

Ba lớp cuối file (`FramePrefetcher`, `AsyncVideoWriter`, `StageProfiler`) là phần
tăng tốc vòng lặp frame: đọc/ghi/đo không chặn luồng chính.
"""

from __future__ import annotations

import math
import os
import queue
import shutil
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterable

import cv2
import numpy as np
import torch

# `ultralytics.utils.nms.non_max_suppression` chỉ dùng `torchvision.ops.nms`
# khi torchvision đã nằm trong sys.modules; nếu không nó rơi về `TorchNMS.nms`
# (chậm hơn ~7x trên tensor nhỏ và **khác thứ tự tie-break**). Ultralytics tự
# import torchvision lúc dựng YOLO, nên đường fast phải import ở đây để kết quả
# trùng khít bit-for-bit với đường Ultralytics.
try:  # pragma: no cover - phụ thuộc môi trường
    import torchvision  # noqa: F401
except Exception:  # pragma: no cover
    pass

# Kernel mặc định đồng bộ notebook4234ae1bfe (ANON_KSIZE = 15).
DEFAULT_BLUR_KERNEL = 15


# =============================================================================
# Thiết bị / cảnh báo CUDA
# =============================================================================


def nvidia_smi_available() -> bool:
    """Máy có driver NVIDIA hay không (không phụ thuộc PyTorch)."""
    return shutil.which("nvidia-smi") is not None


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def resolve_device() -> str:
    """Chọn thiết bị chạy: ưu tiên `NCKH_DEVICE`, sau đó `cuda:0` nếu có CUDA."""
    forced = os.getenv("NCKH_DEVICE", "").strip()
    if forced:
        if forced.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"NCKH_DEVICE={forced} nhưng torch.cuda.is_available() = False. "
                "Cài lại PyTorch bản CUDA (xem requirements-cuda.txt) hoặc bỏ NCKH_DEVICE."
            )
        return forced
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def gpu_name() -> str:
    if not torch.cuda.is_available():
        return "N/A"
    try:
        return str(torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception:
        return "N/A"


def runtime_info() -> dict[str, Any]:
    """Thông tin runtime để ghi vào log/CSV (không dùng để quyết định gì)."""
    try:
        device = resolve_device()
    except Exception:
        device = "cpu"
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda) if torch.version.cuda else "None",
        "cuda_available": cuda_available(),
        "gpu_name": gpu_name(),
        "device": device,
        "cv_threads": int(cv2.getNumThreads()),
        "torch_threads": int(torch.get_num_threads()),
    }


def warn_if_cuda_missing(tag: str = "[NCKH]") -> bool:
    """Cảnh báo to khi máy có NVIDIA nhưng PyTorch không thấy CUDA.

    Trả True nếu đã in cảnh báo. Mục đích là để người chạy không hiểu nhầm
    "GPU chậm" trong khi thực tế pipeline đang chạy CPU.
    """
    if torch.cuda.is_available():
        return False
    if not nvidia_smi_available():
        return False
    lines = [
        "=" * 72,
        f"{tag} CẢNH BÁO: máy có GPU NVIDIA (nvidia-smi tồn tại) nhưng PyTorch",
        f"{tag} KHÔNG thấy CUDA -> pipeline đang chạy CPU, FPS sẽ thấp hơn nhiều.",
        f"{tag} torch={torch.__version__} | torch.version.cuda={torch.version.cuda}",
        f"{tag} Nguyên nhân thường gặp: đã cài requirements.txt (bản '+cpu').",
        f"{tag} Cách sửa (chọn chỉ mục CUDA khớp driver, ví dụ cu124):",
        f"{tag}   pip uninstall -y torch torchvision onnxruntime",
        f"{tag}   pip install --index-url https://download.pytorch.org/whl/cu124 "
        "torch==2.15.0+cu124 torchvision==0.29.2+cu124",
        f"{tag}   pip install -r requirements-cuda.txt",
        f"{tag} Kiểm tra lại: python -c \"import torch; print(torch.cuda.is_available())\"",
        "=" * 72,
    ]
    print("\n".join(lines), flush=True)
    return True


def resolve_half(backend: str, device: str) -> bool:
    """Quyết định dùng FP16 cho detector.

    Mặc định bật khi (backend pytorch + thiết bị CUDA). Tắt bằng `NCKH_HALF=0`.
    Backend ONNX không dùng cờ này vì model ONNX đã cố định precision.
    """
    env = os.getenv("NCKH_HALF", "").strip()
    if env != "":
        return env not in {"0", "false", "False", "no", "NO"}
    return backend == "pytorch" and str(device).startswith("cuda")


# =============================================================================
# Thread
# =============================================================================


def configure_threads() -> dict[str, int]:
    """Áp dụng cấu hình thread cho OpenCV và PyTorch.

    - `NCKH_CV_THREADS`: đặt thẳng cho OpenCV (0 = tuần tự, hữu ích khi đo).
      **Không đặt thì mặc định bật `min(4, số CPU)`**: `import ultralytics` gọi
      `cv2.setNumThreads(0)` nên nếu để nguyên, GaussianBlur của blur/SSIM chạy
      1 thread và chậm 1,5-2x. Đây chỉ là thay đổi hiệu năng, không đổi pixel.
    - `NCKH_TORCH_THREADS`: đặt thẳng cho PyTorch. Nếu không đặt và đang chạy
      CUDA, giới hạn còn min(8, số CPU) để tránh tranh chấp CPU với GPU.
    """
    env_cv = os.getenv("NCKH_CV_THREADS", "").strip()
    if env_cv != "":
        cv2.setNumThreads(int(env_cv))
    else:
        cv2.setNumThreads(max(1, min(4, os.cpu_count() or 1)))

    env_torch = os.getenv("NCKH_TORCH_THREADS", "").strip()
    if env_torch != "":
        torch.set_num_threads(max(1, int(env_torch)))
    elif torch.cuda.is_available():
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    return {
        "cv_threads": int(cv2.getNumThreads()),
        "torch_threads": int(torch.get_num_threads()),
    }


# =============================================================================
# Blur theo ROI
# =============================================================================


def _clamp_box(box: Iterable[float], w: int, h: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [int(v) for v in list(box)[:4]]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    """Chuyển mask uint8 0/1 sang bool không copy (view cùng itemsize)."""
    if mask.dtype == np.bool_:
        return mask
    if mask.dtype == np.uint8:
        return mask.view(np.bool_)
    return mask.astype(bool)


def roi_gaussian_blur(
    frame: np.ndarray,
    boxes: list[list[float]],
    ksize: int = DEFAULT_BLUR_KERNEL,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Blur theo ROI, cho kết quả **giống hệt** blur toàn frame trong vùng che.

    Cách làm: cắt từng cửa sổ box + margin `ksize // 2`, ghép các cửa sổ thành
    một "mosaic" (xếp kệ), gọi `cv2.GaussianBlur` **một lần** trên mosaic rồi
    dán phần trong box trở lại frame.

    Vì sao tương đương: mỗi box nằm cách biên cửa sổ đúng `half = ksize // 2`
    pixel, nên toàn bộ kernel của mọi pixel trong box nằm trong cửa sổ và đọc
    đúng pixel ảnh gốc. Các cửa sổ không đọc chéo sang nhau vì pixel trong box
    cách biên >= half. Riêng cạnh bị cắt bởi biên ảnh được pad bằng
    `BORDER_REFLECT_101` — đúng phép phản chiếu mà blur toàn frame dùng tại
    biên ảnh.

    Vì sao nhanh hơn: blur toàn frame 1360x765 kernel 15 mất ít ms hơn, còn tổng
    diện tích các cửa sổ chỉ ~35% frame và chỉ có 1 lời gọi OpenCV thay vì N
    lời gọi nhỏ (mỗi lời gọi có phí cố định ~0.5-1 ms).

    Nếu tổng diện tích cửa sổ quá lớn (frame quá đông người) thì tự fallback về
    blur toàn frame để không chậm hơn bản cũ.

    `mask` (uint8/bool, HxW) nếu có sẽ giới hạn pixel được ghi đè, đúng bằng
    hành vi `out[mask] = blurred[mask]` của bản cũ.
    """
    h, w = frame.shape[:2]
    if ksize % 2 == 0:
        ksize += 1
    half = ksize // 2

    clamped: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for box in boxes or []:
        c = _clamp_box(box, w, h)
        if c is None or c in seen:
            continue
        seen.add(c)
        clamped.append(c)
    if not clamped:
        return frame.copy()

    mask_view = None if mask is None else _as_bool_mask(mask)
    # `cv2.copyTo` (C++, SIMD) thay cho fancy indexing `out[mask] = src[mask]`.
    # Fancy indexing trên frame 1344x756 với mask ~60% diện tích tốn ~28 ms,
    # `copyTo` cùng việc chỉ ~0.9 ms — đây là chi phí ẩn lớn nhất của blur.
    mask_u8 = None if mask_view is None else mask_view.view(np.uint8)

    # 1) Chuẩn bị tile: cửa sổ box + margin, pad phản chiếu ở cạnh chạm biên ảnh.
    tiles: list[tuple[np.ndarray, tuple[int, int, int, int], int, int, int, int]] = []
    total_area = 0
    max_tile_w = 1
    for x1, y1, x2, y2 in clamped:
        if mask_view is not None and not mask_view[y1:y2, x1:x2].any():
            continue
        cx1 = max(0, x1 - half)
        cy1 = max(0, y1 - half)
        cx2 = min(w, x2 + half)
        cy2 = min(h, y2 + half)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        # Chỉ pad phía thật sự bị biên ảnh cắt (phía còn lại đủ margin thật).
        pad_l = half if (cx1 == 0 and x1 < half) else 0
        pad_t = half if (cy1 == 0 and y1 < half) else 0
        pad_r = half if (cx2 == w and (w - x2) < half) else 0
        pad_b = half if (cy2 == h and (h - y2) < half) else 0
        # BORDER_REFLECT_101 cần pad < kích thước chiều tương ứng.
        pad_l = min(pad_l, crop.shape[1] - 1)
        pad_r = min(pad_r, crop.shape[1] - 1)
        pad_t = min(pad_t, crop.shape[0] - 1)
        pad_b = min(pad_b, crop.shape[0] - 1)
        if pad_l or pad_t or pad_r or pad_b:
            tile = cv2.copyMakeBorder(
                crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101
            )
        else:
            tile = crop
        tiles.append(
            (tile, (x1, y1, x2, y2), pad_l, pad_t, x1 - cx1, y1 - cy1)
        )
        total_area += int(tile.shape[0]) * int(tile.shape[1])
        max_tile_w = max(max_tile_w, int(tile.shape[1]))

    if not tiles:
        return frame.copy()

    # 2) Fallback: frame quá đông hoặc box quá nhiều -> blur toàn frame (chính xác
    #    tuyệt đối, tránh trường hợp mosaic còn lớn hơn cả frame).
    if total_area > 0.6 * h * w or len(tiles) > 256:
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
        out = frame.copy()
        if mask_view is None:
            for x1, y1, x2, y2 in clamped:
                out[y1:y2, x1:x2] = blurred[y1:y2, x1:x2]
        else:
            cv2.copyTo(blurred, mask_u8, out)
        return out

    # 3) Xếp kệ vào mosaic, chọn chiều rộng ~sqrt(diện tích) để ít lãng phí.
    target_w = max(max_tile_w, int(math.sqrt(max(total_area, 1)) * 1.2))
    positions: list[tuple[int, int]] = []
    x = 0
    y = 0
    row_h = 0
    for tile, _box, _pl, _pt, _bx, _by in tiles:
        th, tw = int(tile.shape[0]), int(tile.shape[1])
        if x + tw > target_w and x > 0:
            y += row_h
            x = 0
            row_h = 0
        positions.append((x, y))
        x += tw
        row_h = max(row_h, th)
    mosaic_h = y + row_h
    mosaic = np.empty((mosaic_h, target_w, frame.shape[2]), dtype=np.uint8)
    for (tile, *_rest), (ox, oy) in zip(tiles, positions):
        th, tw = int(tile.shape[0]), int(tile.shape[1])
        mosaic[oy:oy + th, ox:ox + tw] = tile

    # 4) Một lời gọi blur duy nhất, rồi dán phần trong box trở lại.
    blurred = cv2.GaussianBlur(mosaic, (ksize, ksize), 0)
    out = frame.copy()
    for (tile, (x1, y1, x2, y2), pad_l, pad_t, bx1, by1), (ox, oy) in zip(
        tiles, positions
    ):
        sy = oy + pad_t + by1
        sx = ox + pad_l + bx1
        src = blurred[sy:sy + (y2 - y1), sx:sx + (x2 - x1)]
        dst = out[y1:y2, x1:x2]
        if mask_view is None:
            dst[:] = src
        else:
            cv2.copyTo(src, mask_u8[y1:y2, x1:x2], dst)
    return out


# =============================================================================
# Đường ống I/O bất đồng bộ cho vòng lặp frame
# =============================================================================


def _imread_bgr(path: Any) -> np.ndarray | None:
    """Đọc JPEG ở chế độ BGR; tách riêng để thread có thể gọi trực tiếp."""
    return cv2.imread(str(path))


# =============================================================================
# Detect theo lưới tile (dùng chung Bước 5 và Bước 6)
# =============================================================================


def parse_tiles(spec: str) -> tuple[int, int]:
    """Đọc lưới tile dạng '2x2' -> (cols, rows). '1x1' hoặc rỗng -> tắt tiling."""
    text = str(spec or "").strip().lower().replace("*", "x")
    if not text or text in {"1x1", "none", "off", "0"}:
        return (1, 1)
    if "x" not in text:
        raise ValueError(f"det_tiles không hợp lệ: {spec!r} (mong đợi dạng '2x2')")
    cols_s, rows_s = text.split("x", 1)
    cols, rows = int(cols_s), int(rows_s)
    if cols < 1 or rows < 1:
        raise ValueError(f"det_tiles không hợp lệ: {spec!r}")
    return (cols, rows)


def nms_detections(det: np.ndarray, iou_thres: float, max_det: int) -> np.ndarray:
    """NMS trên mảng (N, 6) = x1,y1,x2,y2,conf,cls. Vector hoá bằng OpenCV."""
    det = np.asarray(det, dtype=np.float32).reshape(-1, 6)
    if len(det) == 0:
        return np.empty((0, 6), dtype=np.float32)
    x1, y1, x2, y2 = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
    boxes_wh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    idx = cv2.dnn.NMSBoxes(
        boxes_wh, det[:, 4].astype(np.float32).tolist(), 0.0, float(iou_thres)
    )
    if idx is None or len(idx) == 0:
        return np.empty((0, 6), dtype=np.float32)
    idx = np.asarray(idx).reshape(-1)[: int(max_det)]
    return det[idx].astype(np.float32)


def run_tiled_detect(
    model: Any,
    frame: np.ndarray,
    img_size: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    tiles: str = "1x1",
    overlap: float = 0.15,
    predict_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Chạy detector trên toàn frame hoặc trên lưới tile rồi gộp bằng NMS.

    Vì sao: VisDrone có rất nhiều người nhỏ. Đưa nguyên frame 1344x756 vào
    `imgsz=1024` làm scale còn 0,76 -> mất chi tiết. Tiling giữ scale gần native
    (2x2 @640 -> ~0,83) mà vẫn rẻ hơn chạy `imgsz=1344` (1,56x so với 1,72x).

    Trả về mảng (N, 6): x1, y1, x2, y2, conf, cls (toạ độ frame gốc).
    """
    kwargs = dict(predict_kwargs or {})
    cols, rows = parse_tiles(tiles)
    if cols == 1 and rows == 1:
        res = model(
            frame, imgsz=img_size, conf=conf_thres, iou=iou_thres,
            max_det=max_det, **kwargs,
        )[0]
        boxes = res.cpu().boxes
        xyxy = boxes.xyxy.numpy() if len(boxes) else np.empty((0, 4), np.float32)
        confs = boxes.conf.numpy() if len(boxes) else np.empty((0,), np.float32)
        if not len(xyxy):
            return np.empty((0, 6), dtype=np.float32)
        return np.column_stack([xyxy, confs, np.zeros_like(confs)]).astype(np.float32)

    h, w = frame.shape[:2]
    tile_w = w / cols * (1.0 + float(overlap))
    tile_h = h / rows * (1.0 + float(overlap))
    xs = np.linspace(0.0, max(w - tile_w, 0.0), cols) if w > tile_w else np.zeros(cols)
    ys = np.linspace(0.0, max(h - tile_h, 0.0), rows) if h > tile_h else np.zeros(rows)

    chunks: list[np.ndarray] = []
    for y0 in ys:
        for x0 in xs:
            crop = frame[int(y0): int(y0 + tile_h), int(x0): int(x0 + tile_w)]
            if crop.size == 0:
                continue
            res = model(
                crop, imgsz=img_size, conf=conf_thres, iou=iou_thres,
                max_det=max(100, int(max_det)), **kwargs,
            )[0]
            boxes = res.cpu().boxes
            if not len(boxes):
                continue
            xyxy = boxes.xyxy.numpy().astype(np.float32)
            confs = boxes.conf.numpy().astype(np.float32)
            offset = np.array([x0, y0, x0, y0], dtype=np.float32)
            chunks.append(
                np.column_stack([xyxy + offset, confs, np.zeros_like(confs)])
                .astype(np.float32)
            )
    if not chunks:
        return np.empty((0, 6), dtype=np.float32)
    # NMS gộp toàn cục để khử box trùng ở vùng chồng lấn giữa các tile.
    return nms_detections(np.vstack(chunks), iou_thres, max_det)


class FastOpenVINODetector:
    """Detector OpenVINO tối giản, tương đương bit-for-bit với Ultralytics.

    Vì sao cần: `run_tiled_detect(..., tiles="1x1")` đi qua predictor của
    Ultralytics, đo được ~51 ms **CPU** mỗi frame. Phần lớn chi phí đó là NMS
    chạy 4 thread trên tensor rất nhỏ: mỗi phép toán torch phải đồng bộ thread
    pool, nên NMS tốn ~14 ms wall / ~53 ms CPU thay vì ~2 ms khi chạy 1 thread.
    Khi detect chạy ở thread riêng (overlap với blur/mask), lượng CPU đó tranh
    chấp với luồng chính và làm blur 5 ms phình lên ~48 ms.

    Ở đây: letterbox + chuẩn hoá bằng numpy (đúng công thức `LetterBox` của
    Ultralytics, đã đối chứng cho kết quả trùng khít), suy luận bằng
    `CompiledModel` gọi trực tiếp (bỏ lớp Python của predictor), rồi NMS của
    Ultralytics với 1 thread. CPU mỗi frame còn ~18 ms.

    Chỉ dùng cho `det_tiles == "1x1"`; tiling vẫn đi qua `run_tiled_detect`.
    """

    def __init__(self, model_dir: Any, device: str, img_size: int) -> None:
        from pathlib import Path

        import openvino as ov

        xml = next(iter(sorted(Path(model_dir).glob("*.xml"))), None)
        if xml is None:
            raise FileNotFoundError(f"Không thấy file .xml trong {model_dir}")
        dev = str(device)
        if dev.startswith("intel:"):
            dev = dev[len("intel:") :]
        self.device = dev
        self.img_size = int(img_size)
        self._core = ov.Core()
        # Dùng đúng config của Ultralytics (`AutoBackend`): PERFORMANCE_HINT
        # khác nhau chọn kernel khác nhau, cho kết quả lệch ~1e-7 -> NMS có thể
        # đổi số box ở ngưỡng conf thấp (0.05). Giữ nguyên để tương đương bit.
        self._compiled = self._core.compile_model(
            str(xml), dev, config={"PERFORMANCE_HINT": "LATENCY"}
        )
        self._in_port = self._compiled.input(0)
        self._out_port = self._compiled.output(0)

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float, int, int]:
        """Letterbox y hệt `ultralytics.data.augment.LetterBox` khi predict.

        Trả `(tensor, gain_x, gain_y, left, top)`. Lưu ý `gain_x != gain_y` khi
        ảnh bị phóng to: `scale_boxes` của Ultralytics chia trục x cho
        `new_w / w` và trục y cho `new_h / h` (không dùng chung một `r`), nên
        phải trả cả hai để toạ độ khớp bit-for-bit.
        """
        size = self.img_size
        h, w = frame.shape[:2]
        r = min(size / h, size / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = (
            cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            if (nw, nh) != (w, h)
            else frame
        )
        dw, dh = (size - nw) / 2.0, (size - nh) / 2.0
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        # BGR->RGB, HWC->CHW, float32/255 (đúng thứ tự predictor của Ultralytics).
        x = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
        x /= 255.0
        return x[None], nw / w, nh / h, left, top

    def __call__(
        self,
        frame: np.ndarray,
        conf_thres: float,
        iou_thres: float,
        max_det: int,
    ) -> np.ndarray:
        from ultralytics.utils.nms import non_max_suppression

        h, w = frame.shape[:2]
        x, gain_x, gain_y, left, top = self._preprocess(frame)
        pred = np.array(self._compiled(x)[self._out_port], copy=True)
        det = non_max_suppression(
            torch.from_numpy(pred), conf_thres, iou_thres,
            max_det=int(max_det), nc=1,
        )[0]
        if det is None or len(det) == 0:
            return np.empty((0, 6), dtype=np.float32)
        d = det.numpy()
        # Đưa box từ hệ toạ độ letterbox về frame gốc (giống `scale_boxes`).
        d[:, [0, 2]] -= left
        d[:, [1, 3]] -= top
        d[:, [0, 2]] /= gain_x
        d[:, [1, 3]] /= gain_y
        d[:, [0, 2]] = d[:, [0, 2]].clip(0, w)
        d[:, [1, 3]] = d[:, [1, 3]].clip(0, h)
        confs = d[:, 4].astype(np.float32)
        return np.column_stack(
            [d[:, :4].astype(np.float32), confs, np.zeros_like(confs)]
        ).astype(np.float32)


class FramePrefetcher:
    """Đọc trước JPEG bằng thread, trả frame **đúng thứ tự** file.

    Vì sao nhanh: `cv2.imread` nhả GIL nên nhiều thread giải mã thật sự song
    song; luồng chính không còn chờ I/O + giải mã mỗi frame.

    Vì sao không đổi kết quả: mỗi frame vẫn được đọc từ đúng file theo đúng thứ
    tự (giống hệt vòng `for` tuần tự), chỉ thời điểm đọc là sớm hơn.

    Bộ nhớ bị chặn bởi `max_pending`: chỉ tối đa chừng đó frame được giữ trong
    RAM (1360x765x3 ~ 3 MB/frame).
    """

    def __init__(
        self,
        paths: Iterable[Any],
        workers: int = 3,
        max_pending: int = 16,
    ) -> None:
        self._paths = list(paths)
        self._max_pending = max(1, int(max_pending))
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self._pending: deque[tuple[Any, Future]] = deque()
        self._paths_iter = iter(self._paths)
        self._exhausted = False
        # Thống kê thời gian luồng chính phải chờ dữ liệu (0 = prefetch đủ nhanh).
        self.wait_ms = 0.0
        self.frames_read = 0
        self._fill()

    def _fill(self) -> None:
        while not self._exhausted and len(self._pending) < self._max_pending:
            try:
                path = next(self._paths_iter)
            except StopIteration:
                self._exhausted = True
                break
            self._pending.append((path, self._executor.submit(_imread_bgr, path)))

    def __iter__(self) -> "FramePrefetcher":
        return self

    def __next__(self) -> tuple[Any, np.ndarray | None]:
        if not self._pending:
            raise StopIteration
        path, future = self._pending.popleft()
        t0 = time.perf_counter()
        frame = future.result()
        self.wait_ms += (time.perf_counter() - t0) * 1000.0
        self.frames_read += 1
        self._fill()
        return path, frame

    def close(self) -> None:
        """Dừng executor; gọi sau khi vòng lặp kết thúc (kể cả khi lỗi)."""
        self._executor.shutdown(wait=True)
        self._pending.clear()

    def __del__(self) -> None:  # pragma: no cover - dọn dẹp khi có exception
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    def __enter__(self) -> "FramePrefetcher":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class AsyncVideoWriter:
    """Ghi video trong thread riêng, cùng interface `write`/`release`.

    `write()` chỉ đẩy frame vào queue; khi queue đầy thì **chặn** (backpressure)
    để bộ nhớ không tăng vô hạn và để FPS phản ánh đúng thông lượng thật, không
    phải "ẩn" chi phí encode.

    Không copy frame: caller không được sửa frame sau khi gọi `write()` (step5
    tạo frame mới cho mỗi lần ghi nên an toàn).
    """

    def __init__(
        self,
        path: Any,
        fourcc: int,
        fps: float,
        size: tuple[int, int],
        max_pending: int = 8,
    ) -> None:
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        self._opened = bool(self._writer.isOpened())
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(max_pending)))
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        if self._opened:
            self._thread = threading.Thread(
                target=self._run, name="video-writer", daemon=False
            )
            self._thread.start()
        else:
            self._writer.release()

    def isOpened(self) -> bool:
        return self._opened

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            frame, render = item
            if self._error is not None:
                continue
            try:
                self._writer.write(render(frame) if render is not None else frame)
            except BaseException as exc:  # pragma: no cover - phụ thuộc codec
                self._error = exc

    def write(self, frame: np.ndarray, render: Any = None) -> None:
        """Đẩy frame (và hàm render tuỳ chọn) vào hàng đợi của thread ghi."""
        if self._opened:
            self._queue.put((frame, render))

    def release(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None
        self._writer.release()
        if self._error is not None:
            # Video demo chỉ để minh hoạ: không được làm mất metric của cả run,
            # nên cảnh báo rồi tiếp tục thay vì raise.
            print(
                f"[perf_utils] Cảnh báo: ghi video lỗi ({self._error}); "
                "video demo có thể thiếu frame.",
                flush=True,
            )


class StageProfiler:
    """Đo thời gian từng stage của vòng lặp frame (bật bằng `NCKH_PROFILE=1`).

    Khi tắt, `tick()` trả 0.0 và `mark()` không làm gì (chi phí ~1 lệnh if).
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._samples: dict[str, list[float]] = {}
        self._totals: dict[str, list[float]] = {}

    def tick(self) -> float:
        return time.perf_counter() if self.enabled else 0.0

    def mark(self, name: str, t0: float) -> None:
        if not self.enabled or t0 <= 0.0:
            return
        self._samples.setdefault(name, []).append(
            (time.perf_counter() - t0) * 1000.0
        )

    def add(self, name: str, ms: float) -> None:
        if self.enabled:
            self._samples.setdefault(name, []).append(float(ms))

    def add_total(self, name: str, total_ms: float, count: int) -> None:
        """Ghi một tổng đã cộng sẵn (ví dụ thời gian chờ decode của prefetcher)."""
        if self.enabled and count > 0:
            self._totals.setdefault(name, []).append(float(total_ms) / count)

    def merge(self, other: "StageProfiler") -> None:
        if not self.enabled or not other.enabled:
            return
        for name, values in other._samples.items():
            self._samples.setdefault(name, []).extend(values)
        for name, values in other._totals.items():
            self._totals.setdefault(name, []).extend(values)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, values in self._samples.items():
            if values:
                out[name] = _describe(values)
        for name, values in self._totals.items():
            if values:
                out[name] = _describe(values)
        return out

    def format_table(self) -> str:
        summary = self.summary()
        if not summary:
            return ""
        rows = [
            f"{'stage':<16}{'n':>7}{'mean':>10}{'p50':>9}{'p95':>9}{'max':>9}"
        ]
        for name, s in summary.items():
            rows.append(
                f"{name:<16}{int(s['count']):>7}{s['mean']:>10.2f}"
                f"{s['p50']:>9.2f}{s['p95']:>9.2f}{s['max']:>9.2f}"
            )
        return "\n".join(rows)


def _describe(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }
