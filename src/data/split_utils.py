"""
Split utilities for creating train/val/test splits.
Splits are identity-aware to prevent data leakage.
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd


def get_identity_map(coco: dict) -> Dict[str, Set[int]]:
    """
    Map identity_id -> set of image_ids.
    If no identity info, falls back to treating each image as its own identity.
    """
    identity_to_images = defaultdict(set)

    for ann in coco["annotations"]:
        attrs = ann.get("attributes", {})
        identity = attrs.get("identity_id", f"auto_{ann['image_id']}")
        if identity == "NA":
            identity = f"auto_{ann['image_id']}"
        identity_to_images[identity].add(ann["image_id"])

    return dict(identity_to_images)


def split_by_identity(
    coco: dict,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Split image IDs into train/val/test sets by identity.
    No identity appears in more than one split.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    identity_map = get_identity_map(coco)
    identities = sorted(identity_map.keys())

    random.seed(seed)
    random.shuffle(identities)

    n = len(identities)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = identities[:n_train]
    val_ids = identities[n_train:n_train + n_val]
    test_ids = identities[n_train + n_val:]

    splits = {"train": [], "val": [], "test": []}
    for id_name in train_ids:
        splits["train"].extend(identity_map[id_name])
    for id_name in val_ids:
        splits["val"].extend(identity_map[id_name])
    for id_name in test_ids:
        splits["test"].extend(identity_map[id_name])

    # Sort for reproducibility
    for k in splits:
        splits[k] = sorted(set(splits[k]))

    return splits


def compute_dataset_stats(coco: dict, splits: Dict[str, List[int]]) -> pd.DataFrame:
    """Compute basic statistics about the dataset."""
    # Bbox sizes
    bbox_areas = [ann["area"] for ann in coco["annotations"]]
    bbox_widths = [ann["bbox"][2] for ann in coco["annotations"]]
    bbox_heights = [ann["bbox"][3] for ann in coco["annotations"]]

    # Image sizes
    img_widths = [img["width"] for img in coco["images"]]
    img_heights = [img["height"] for img in coco["images"]]

    # Identity count
    identity_map = get_identity_map(coco)

    stats = {
        "total_images": len(coco["images"]),
        "total_annotations": len(coco["annotations"]),
        "total_categories": len(coco["categories"]),
        "unique_identities": len(identity_map),
        "train_images": len(splits.get("train", [])),
        "val_images": len(splits.get("val", [])),
        "test_images": len(splits.get("test", [])),
        "avg_bbox_area": sum(bbox_areas) / max(len(bbox_areas), 1),
        "avg_bbox_width": sum(bbox_widths) / max(len(bbox_widths), 1),
        "avg_bbox_height": sum(bbox_heights) / max(len(bbox_heights), 1),
        "min_bbox_area": min(bbox_areas) if bbox_areas else 0,
        "max_bbox_area": max(bbox_areas) if bbox_areas else 0,
        "avg_img_width": sum(img_widths) / max(len(img_widths), 1),
        "avg_img_height": sum(img_heights) / max(len(img_heights), 1),
        "avg_annotations_per_image": len(coco["annotations"]) / max(len(coco["images"]), 1),
    }

    return pd.DataFrame([stats]).T.rename(columns={0: "value"})


def save_splits(splits: Dict[str, List[int]], output_path: str) -> None:
    """Save splits to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits saved to {output_path}")
    for k, v in splits.items():
        print(f"  {k}: {len(v)} images")
