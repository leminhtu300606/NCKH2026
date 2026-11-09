"""Pipeline chạy từ Bước 3 đến Bước 6 (dựa trên artifacts đã có trong file zip)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

STEPS = [
    ("Bước 5: Tích hợp hệ thống hoàn chỉnh (test-dev)", "step5_integrated_system.py"),
    ("Bước 6: Đo hiệu năng phần cứng", "step6_hardware_benchmark.py"),
]


def run_step(name: str, script: str) -> int:
    print(f"\n{'='*60}")
    print(f"Đang chạy: {name}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, script], cwd=Path(__file__).parent)
    return result.returncode


def main() -> None:
    for name, script in STEPS:
        rc = run_step(name, script)
        if rc != 0:
            print(f"\nLỖI: {name} kết thúc với mã {rc}")
            sys.exit(rc)
    print("\n" + "=" * 60)
    print("TẤT CẢ CÁC BƯỚC ĐÃ HOÀN TẤT")
    print("=" * 60)
    print("Kết quả chính:")
    print(" - results/system_metrics_unified.csv   (chỉ số đánh giá hệ thống, test-dev)")
    print(" - results/val_tracking_metrics.csv     (chỉ số tracking, valset)")
    print(" - results/hardware_benchmark.csv       (hiệu năng phần cứng)")
    print(" - results/output_gate_log.json         (nhật ký chặn đầu ra)")
    print(" - outputs/protected_video.mp4          (video đã che đối tượng)")


if __name__ == "__main__":
    main()
