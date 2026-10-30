#!/usr/bin/env python3
"""
Export trained detector to ONNX format.

Usage:
    python scripts/export_onnx.py --weights artifacts/detector/best.pt --img-size 640
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.detector import PersonDetector


def main():
    parser = argparse.ArgumentParser(description="Export detector to ONNX")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
    parser.add_argument("--config", type=str, default="configs/detector.yaml")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX path")
    parser.add_argument("--img-size", type=int, default=640)
    args = parser.parse_args()

    detector = PersonDetector(config_path=args.config, weights=args.weights)
    onnx_path = detector.export_onnx(output_path=args.output)
    print(f"\n✅ ONNX model exported: {onnx_path}")
    print(f"   Size: {os.path.getsize(onnx_path) / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
