#!/usr/bin/env python3
"""
Phase 1 — Train Person Detector
================================

Train a single-class person detector using Ultralytics YOLO.

Usage (Kaggle):
    !python scripts/train_detector.py --data /kaggle/working/data/processed/yolo_dataset/dataset.yaml

Usage (local):
    python scripts/train_detector.py --data data/processed/yolo_dataset/dataset.yaml --config configs/detector.yaml
"""

import argparse
import os
import sys

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.detector import PersonDetector


def main():
    parser = argparse.ArgumentParser(description="Train person detector")
    parser.add_argument("--data", type=str, required=True, help="Path to YOLO dataset.yaml")
    parser.add_argument("--config", type=str, default="configs/detector.yaml", help="Detector config YAML")
    parser.add_argument("--weights", type=str, default=None, help="Resume from weights")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    args = parser.parse_args()

    # Initialize detector
    detector = PersonDetector(config_path=args.config, weights=args.weights)

    if not args.eval_only:
        # ─── Train ──────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("PHASE 1: Training person detector")
        print("=" * 70)

        results = detector.train(data_yaml=args.data)
        print("\n✅ Training complete!")

        # Reload best weights
        best_pt = os.path.join(detector.paths["checkpoint_dir"], "weights", "best.pt")
        if os.path.isfile(best_pt):
            detector = PersonDetector(config_path=args.config, weights=best_pt)
            print(f"Loaded best weights from {best_pt}")

    # ─── Evaluate ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Evaluating detector")
    print("=" * 70)

    eval_results = detector.evaluate(data_yaml=args.data)
    print(f"\nmAP@0.5:      {eval_results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {eval_results.box.map:.4f}")

    # ─── Benchmark ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Benchmarking latency")
    print("=" * 70)

    # Create a dummy frame for benchmarking
    img_size = detector.model_cfg["input_size"]
    dummy_frame = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    bench = detector.benchmark(dummy_frame, n_runs=50)
    print(f"Mean latency: {bench['mean_ms']:.1f} ms")
    print(f"P50 latency:  {bench['p50_ms']:.1f} ms")
    print(f"P95 latency:  {bench['p95_ms']:.1f} ms")
    print(f"FPS:          {bench['fps']:.1f}")
    print(f"Model size:   {bench['model_size_mb']:.1f} MB")

    # ─── Generate report ────────────────────────────────────────────────
    report_path = detector.generate_eval_report(eval_results.box, bench)
    print(f"\n✅ Report saved to {report_path}")

    # ─── Export ONNX ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Exporting ONNX")
    print("=" * 70)

    onnx_path = detector.export_onnx()
    print(f"✅ ONNX exported to {onnx_path}")

    # ─── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"  Best weights: {detector.paths['best_pt']}")
    print(f"  ONNX model:   {onnx_path}")
    print(f"  Eval report:  {report_path}")
    print(f"\nNext: run Phase 2 (tracking) with the trained detector.")


if __name__ == "__main__":
    main()
