#!/usr/bin/env python3
"""
Visualize detector predictions on sample images.
Saves annotated images to artifacts/visuals/.

Usage:
    python scripts/visualize_predictions.py \
        --weights artifacts/detector/best.pt \
        --source data/processed/yolo_dataset/images/val \
        --output artifacts/visuals \
        --max-images 10
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.detector import PersonDetector


def draw_detections(frame, detections, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes and confidence scores on frame."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        conf = det["confidence"]
        label = f"person {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="artifacts/detector/best.pt")
    parser.add_argument("--config", type=str, default="configs/detector.yaml")
    parser.add_argument("--source", type=str, required=True, help="Directory of images")
    parser.add_argument("--output", type=str, default="artifacts/visuals")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence for visualization")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load detector
    detector = PersonDetector(config_path=args.config, weights=args.weights)

    # Get image list
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = [
        os.path.join(args.source, f)
        for f in os.listdir(args.source)
        if os.path.splitext(f)[1].lower() in exts
    ]

    if len(images) > args.max_images:
        random.seed(42)
        images = random.sample(images, args.max_images)

    print(f"Visualizing {len(images)} images...")

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        detections = detector.predict_frame(frame)
        # Filter by visualization confidence
        detections = [d for d in detections if d["confidence"] >= args.conf]

        annotated = draw_detections(frame, detections)
        out_name = os.path.basename(img_path)
        out_path = os.path.join(args.output, f"pred_{out_name}")
        cv2.imwrite(out_path, annotated)
        print(f"  {out_name}: {len(detections)} detections -> {out_path}")

    print(f"\n✅ Saved {len(images)} visualizations to {args.output}")


if __name__ == "__main__":
    main()
