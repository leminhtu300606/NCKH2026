"""
Unit tests for PersonDetector.
Run: pytest tests/test_detector.py -v
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPersonDetector:
    """Tests for the PersonDetector class."""

    def test_config_loading(self):
        """Test that config file loads correctly."""
        import yaml
        config_path = "configs/detector.yaml"
        if not os.path.isfile(config_path):
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "model" in config
        assert "training" in config
        assert "inference" in config
        assert config["model"]["num_classes"] == 1
        assert config["inference"]["conf_threshold"] == 0.01

    def test_predict_frame_output_format(self):
        """Test that predict_frame returns correct format."""
        try:
            from src.detection.detector import PersonDetector
        except ImportError:
            pytest.skip("ultralytics not installed")

        config_path = "configs/detector.yaml"
        if not os.path.isfile(config_path):
            pytest.skip("Config file not found")

        try:
            detector = PersonDetector(config_path=config_path)
        except Exception:
            pytest.skip("Cannot load pretrained model (no network?)")

        # Create dummy image
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        detections = detector.predict_frame(frame)

        assert isinstance(detections, list)
        for det in detections:
            assert "bbox_xyxy" in det
            assert "confidence" in det
            assert "class_id" in det
            assert len(det["bbox_xyxy"]) == 4
            assert 0 <= det["confidence"] <= 1
            assert det["class_id"] == 0  # single class

    def test_benchmark_output_keys(self):
        """Test that benchmark returns expected keys."""
        try:
            from src.detection.detector import PersonDetector
        except ImportError:
            pytest.skip("ultralytics not installed")

        config_path = "configs/detector.yaml"
        if not os.path.isfile(config_path):
            pytest.skip("Config file not found")

        try:
            detector = PersonDetector(config_path=config_path)
        except Exception:
            pytest.skip("Cannot load pretrained model")

        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        bench = detector.benchmark(frame, n_runs=3)

        expected_keys = ["mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "fps", "model_size_mb"]
        for key in expected_keys:
            assert key in bench, f"Missing key: {key}"
            assert isinstance(bench[key], float)


class TestDataConverter:
    """Tests for dataset conversion utilities."""

    def test_find_images(self):
        """Test image finder function."""
        from src.data.dataset_converter import find_images
        # Should not crash on non-existent dir
        result = find_images("/tmp/nonexistent_dir_12345")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_empty_coco(self):
        """Test empty COCO creation."""
        from src.data.dataset_converter import _empty_coco
        coco = _empty_coco(["person"])
        assert len(coco["categories"]) == 1
        assert coco["categories"][0]["name"] == "person"
        assert len(coco["images"]) == 0
        assert len(coco["annotations"]) == 0


class TestSplitUtils:
    """Tests for split utilities."""

    def test_split_by_identity(self):
        """Test identity-based splitting."""
        from src.data.split_utils import split_by_identity

        # Create fake COCO with 10 images, 10 annotations, 5 identities
        coco = {
            "images": [{"id": i} for i in range(1, 11)],
            "annotations": [
                {
                    "image_id": i,
                    "attributes": {"identity_id": f"id_{i % 5}"}
                }
                for i in range(1, 11)
            ],
            "categories": [{"id": 1, "name": "person"}],
        }

        splits = split_by_identity(coco, seed=42)
        all_ids = splits["train"] + splits["val"] + splits["test"]

        # No overlap
        assert len(all_ids) == len(set(all_ids)), "Splits have overlapping image IDs"
        # All images accounted for
        assert set(all_ids) == set(range(1, 11))

    def test_split_ratios(self):
        """Test that split ratios are approximately correct."""
        from src.data.split_utils import split_by_identity

        coco = {
            "images": [{"id": i} for i in range(1, 101)],
            "annotations": [
                {
                    "image_id": i,
                    "attributes": {"identity_id": f"id_{i}"}  # each image = unique identity
                }
                for i in range(1, 101)
            ],
            "categories": [{"id": 1, "name": "person"}],
        }

        splits = split_by_identity(coco, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
        assert len(splits["train"]) == 70
        assert len(splits["val"]) == 15
        assert len(splits["test"]) == 15


class TestYoloConverter:
    """Tests for YOLO format conversion."""

    def test_coco_to_yolo_creates_yaml(self, tmp_path):
        """Test YOLO conversion creates dataset.yaml."""
        from src.data.yolo_converter import coco_to_yolo

        coco = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
            "annotations": [{
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [100, 100, 50, 80], "area": 4000, "iscrowd": 0,
            }],
            "categories": [{"id": 1, "name": "person"}],
        }
        splits = {"train": [1], "val": [], "test": []}

        out_dir = str(tmp_path / "yolo_test")
        yaml_path = coco_to_yolo(coco, splits, out_dir, single_class=True, symlink=False)

        assert os.path.isfile(yaml_path)
        assert os.path.isfile(os.path.join(out_dir, "labels", "train", "00000001.txt"))
