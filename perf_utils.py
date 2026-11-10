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
- `NCKH_BLUR_KERNEL`   : kích thước kernel Gaussian, mặc định 51 (giữ như bản cũ).
- `NCKH_CV_THREADS`    : số thread OpenCV; không đặt thì giữ mặc định của OpenCV.
- `NCKH_TORCH_THREADS` : số thread PyTorch; không đặt thì chỉ giới hạn khi chạy CUDA.
"""

from __future__ import annotations

import math
import os
import shutil
from typing import Any, Iterable

import cv2
import numpy as np
import torch

# Kernel mặc định của bản cũ (step5.blur_boxes dùng (51, 51)).
DEFAULT_BLUR_KERNEL = 51


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
    """Áp dụng cấu hình thread (chỉ khi người dùng đặt biến môi trường).

    - `NCKH_CV_THREADS`: đặt thẳng cho OpenCV (0 = tuần tự, hữu ích khi đo).
    - `NCKH_TORCH_THREADS`: đặt thẳng cho PyTorch. Nếu không đặt và đang chạy
      CUDA, giới hạn còn min(8, số CPU) để tránh tranh chấp CPU với GPU.
    """
    env_cv = os.getenv("NCKH_CV_THREADS", "").strip()
    if env_cv != "":
        cv2.setNumThreads(int(env_cv))

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

    Vì sao nhanh hơn: blur toàn frame 1360x765 kernel 51 mất ~26 ms, còn tổng
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
            out[mask_view] = blurred[mask_view]
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
            m = mask_view[y1:y2, x1:x2]
            dst[m] = src[m]
    return out
