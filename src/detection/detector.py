"""
Person Detector — wraps Ultralytics YOLO for single-class person detection.

Usage:
    detector = PersonDetector(config_path="configs/detector.yaml")
    detector.train()
    results = detector.predict(frame)
    detector.export_onnx()
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


class PersonDetector:
    """Single-class person detector using Ultralytics YOLO."""

    def __init__(self, config_path: str = "configs/detector.yaml", weights: Optional[str] = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model_cfg = self.config["model"]
        self.train_cfg = self.config["training"]
        self.infer_cfg = self.config["inference"]
        self.export_cfg = self.config["export"]
        self.paths = self.config["paths"]

        self.model = None
        self._load_model(weights)

    def _load_model(self, weights: Optional[str] = None):
        """Load or initialize the YOLO model."""
        from ultralytics import YOLO

        if weights and os.path.isfile(weights):
            print(f"Loading weights from {weights}")
            self.model = YOLO(weights)
        elif self.model_cfg.get("pretrained", True):
            arch = self.model_cfg["architecture"]
            print(f"Loading pretrained {arch}")
            self.model = YOLO(f"{arch}.pt")
        else:
            arch = self.model_cfg["architecture"]
            print(f"Initializing {arch} from scratch")
            self.model = YOLO(f"{arch}.yaml")

    def train(self, data_yaml: str) -> dict:
        """
        Train the detector.

        Args:
            data_yaml: path to YOLO dataset.yaml

        Returns:
            Training results dict.
        """
        print(f"\n{'='*60}")
        print("Starting detector training")
        print(f"{'='*60}")

        results = self.model.train(
            data=data_yaml,
            epochs=self.train_cfg["epochs"],
            batch=self.train_cfg["batch_size"],
            imgsz=self.model_cfg["input_size"],
            device=self.train_cfg["device"],
            optimizer=self.train_cfg["optimizer"],
            lr0=self.train_cfg["lr0"],
            lrf=self.train_cfg["lrf"],
            momentum=self.train_cfg["momentum"],
            weight_decay=self.train_cfg["weight_decay"],
            warmup_epochs=self.train_cfg["warmup_epochs"],
            patience=self.train_cfg["patience"],
            seed=self.train_cfg["seed"],
            workers=self.train_cfg["workers"],
            # Augmentation
            mosaic=self.config["augmentation"]["mosaic"],
            mixup=self.config["augmentation"]["mixup"],
            hsv_h=self.config["augmentation"]["hsv_h"],
            hsv_s=self.config["augmentation"]["hsv_s"],
            hsv_v=self.config["augmentation"]["hsv_v"],
            degrees=self.config["augmentation"]["degrees"],
            translate=self.config["augmentation"]["translate"],
            scale=self.config["augmentation"]["scale"],
            fliplr=self.config["augmentation"]["fliplr"],
            erasing=self.config["augmentation"]["erasing"],
            # Output
            project=os.path.dirname(self.paths["checkpoint_dir"]),
            name=os.path.basename(self.paths["checkpoint_dir"]),
            exist_ok=True,
        )
        return results

    def evaluate(self, data_yaml: str) -> dict:
        """Run validation and return metrics."""
        results = self.model.val(
            data=data_yaml,
            imgsz=self.model_cfg["input_size"],
            conf=self.infer_cfg["conf_threshold"],
            iou=self.infer_cfg["iou_threshold"],
            device=self.train_cfg["device"],
        )
        return results

    def predict(
        self,
        source: Any,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> list:
        """
        Run inference on a frame/image/video.

        Args:
            source: image path, numpy array, or video path.
            conf: confidence threshold (default from config).
            iou: NMS IoU threshold (default from config).

        Returns:
            List of ultralytics Results objects.
        """
        conf = conf or self.infer_cfg["conf_threshold"]
        iou = iou or self.infer_cfg["iou_threshold"]

        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            max_det=self.infer_cfg["max_det"],
            verbose=False,
        )
        return results

    def predict_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Run inference on a single frame and return structured detections.

        Returns:
            List of dicts with keys: bbox_xyxy, confidence, class_id
        """
        results = self.predict(frame)
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    det = {
                        "bbox_xyxy": box.xyxy[0].cpu().numpy().tolist(),
                        "confidence": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                    }
                    detections.append(det)
        return detections

    def export_onnx(self, output_path: Optional[str] = None) -> str:
        """Export model to ONNX format."""
        output_path = output_path or self.paths["best_onnx"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.model.export(
            format="onnx",
            imgsz=self.model_cfg["input_size"],
            simplify=self.export_cfg.get("simplify", True),
            dynamic=self.export_cfg.get("dynamic", False),
            half=self.export_cfg.get("half", False),
        )
        print(f"ONNX model exported to {output_path}")
        return output_path

    def benchmark(self, frame: np.ndarray, n_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark inference latency.

        Returns:
            Dict with p50, p95, mean latency in ms, and FPS.
        """
        latencies = []
        # Warmup
        for _ in range(5):
            self.predict(frame)

        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict(frame)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

        latencies.sort()
        return {
            "mean_ms": sum(latencies) / len(latencies),
            "p50_ms": latencies[len(latencies) // 2],
            "p95_ms": latencies[int(len(latencies) * 0.95)],
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
            "fps": 1000 / (sum(latencies) / len(latencies)),
            "model_size_mb": self._get_model_size_mb(),
        }

    def _get_model_size_mb(self) -> float:
        """Get model file size in MB."""
        if hasattr(self.model, "ckpt_path") and self.model.ckpt_path:
            return os.path.getsize(self.model.ckpt_path) / (1024 * 1024)
        return 0.0

    def generate_eval_report(self, metrics: dict, benchmark: dict, output_path: Optional[str] = None) -> str:
        """Generate a markdown evaluation report."""
        output_path = output_path or self.paths["eval_report"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        targets = self.config["targets"]

        report = f"""# Detector Evaluation Report

## Model Configuration
- **Architecture**: {self.model_cfg['architecture']}
- **Input size**: {self.model_cfg['input_size']}
- **Confidence threshold**: {self.infer_cfg['conf_threshold']}
- **Single-class**: person

## Detection Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| mAP@0.5 | {getattr(metrics, 'map50', 'N/A')} | ≥ {targets['map50']} | {'✅' if getattr(metrics, 'map50', 0) >= targets['map50'] else '❌'} |
| mAP@0.5:0.95 | {getattr(metrics, 'map', 'N/A')} | — | — |
| Precision | {getattr(metrics, 'mp', 'N/A')} | — | — |
| Recall | {getattr(metrics, 'mr', 'N/A')} | ≥ {targets['recall']} | {'✅' if getattr(metrics, 'mr', 0) >= targets['recall'] else '❌'} |

## Latency Benchmark
| Metric | Value | Target |
|--------|-------|--------|
| Mean latency | {benchmark['mean_ms']:.1f} ms | — |
| P50 latency | {benchmark['p50_ms']:.1f} ms | — |
| P95 latency | {benchmark['p95_ms']:.1f} ms | — |
| FPS | {benchmark['fps']:.1f} | ≥ {targets['edge_fps']} |
| Model size | {benchmark['model_size_mb']:.1f} MB | ≤ {targets['model_size_mb']} MB |

## Notes
- Confidence threshold set to {self.infer_cfg['conf_threshold']} to prioritize recall (privacy-first).
- False negatives (missed persons) are more critical than false positives.
"""
        with open(output_path, "w") as f:
            f.write(report)

        print(f"Evaluation report saved to {output_path}")
        return output_path
