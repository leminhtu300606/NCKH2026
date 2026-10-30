#!/usr/bin/env python3
"""
Benchmark detector inference latency.

Usage:
    python scripts/benchmark_latency.py --weights artifacts/detector/best.pt --runs 100
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.detector import PersonDetector


def main():
    parser = argparse.ArgumentParser(description="Benchmark detector latency")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/detector.yaml")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--output", type=str, default="artifacts/detector/benchmark.json")
    args = parser.parse_args()

    detector = PersonDetector(config_path=args.config, weights=args.weights)

    # Create dummy frame
    frame = np.random.randint(0, 255, (args.img_size, args.img_size, 3), dtype=np.uint8)

    print(f"Running {args.runs} inference iterations...")
    results = detector.benchmark(frame, n_runs=args.runs)

    print("\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v:.2f}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Benchmark saved to {args.output}")


if __name__ == "__main__":
    main()
