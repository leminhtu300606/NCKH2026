#!/usr/bin/env python3
"""
Phase 0 — Dataset Preparation Script
=====================================
1. Auto-discovers and converts both datasets to COCO JSON.
2. Creates identity-aware train/val/test splits.
3. Converts to YOLO format for detector training.
4. Generates dataset statistics manifest.

Usage (Kaggle):
    !python scripts/prepare_dataset.py --config configs/dataset.yaml

Usage (local):
    python scripts/prepare_dataset.py --config configs/dataset.yaml
"""

import argparse
import json
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_converter import convert_dataset
from src.data.split_utils import split_by_identity, compute_dataset_stats, save_splits
from src.data.yolo_converter import coco_to_yolo


def _merge_coco(coco_list, names):
    """Merge multiple COCO dicts into one, re-indexing IDs."""
    merged = {
        "info": {"description": f"Merged from: {', '.join(names)}"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }
    cat_name_to_id = {}
    cat_id = 1
    for coco in coco_list:
        for cat in coco["categories"]:
            if cat["name"] not in cat_name_to_id:
                cat_name_to_id[cat["name"]] = cat_id
                merged["categories"].append({"id": cat_id, "name": cat["name"]})
                cat_id += 1

    img_id_offset = 0
    ann_id_offset = 0
    for coco in coco_list:
        old_cat_map = {cat["id"]: cat_name_to_id[cat["name"]] for cat in coco["categories"]}
        old_img_map = {}
        for img in coco["images"]:
            new_id = img["id"] + img_id_offset
            old_img_map[img["id"]] = new_id
            merged["images"].append({**img, "id": new_id})
        for ann in coco["annotations"]:
            new_ann = {**ann}
            new_ann["id"] = ann["id"] + ann_id_offset
            new_ann["image_id"] = old_img_map[ann["image_id"]]
            new_ann["category_id"] = old_cat_map.get(ann["category_id"], ann["category_id"])
            merged["annotations"].append(new_ann)
        img_id_offset += len(coco["images"]) + 1
        ann_id_offset += len(coco["annotations"]) + 1
    return merged


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for UAV Privacy pipeline")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml")
    parser.add_argument("--single-class", type=str, default="person")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in ["interim_dir", "processed_dir", "manifests_dir"]:
        os.makedirs(config["output"][key], exist_ok=True)

    all_coco = []
    dataset_names = []

    # ─── Step 1: Convert each dataset to COCO ───
    print("\n" + "=" * 70)
    print("STEP 1: Convert datasets to COCO format")
    print("=" * 70)

    for ds_key, ds_cfg in config["datasets"].items():
        root = ds_cfg["root"]
        output_path = config["coco_output"].get(
            ds_key.replace("-", "_"),
            os.path.join(config["output"]["interim_dir"], f"coco_{ds_key}.json"),
        )
        if not os.path.isdir(root):
            print(f"\n⚠ WARNING: Dataset root not found: {root}")
            print(f"  Skipping {ds_cfg['name']}. Make sure it is mounted.")
            continue

        coco = convert_dataset(root=root, output_path=output_path, dataset_name=ds_cfg["name"])
        all_coco.append(coco)
        dataset_names.append(ds_key)
        print(f"✅ {ds_cfg['name']}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    if not all_coco:
        print("\n❌ No datasets found! Check your paths in configs/dataset.yaml.")
        sys.exit(1)

    # ─── Step 2: Merge datasets ───
    print("\n" + "=" * 70)
    print("STEP 2: Merge datasets")
    print("=" * 70)

    merged = all_coco[0] if len(all_coco) == 1 else _merge_coco(all_coco, dataset_names)
    merged_path = config["coco_output"].get(
        "merged", os.path.join(config["output"]["interim_dir"], "coco_merged.json"))
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"✅ Merged COCO: {len(merged['images'])} images, {len(merged['annotations'])} annotations")

    # ─── Step 3: Create splits ───
    print("\n" + "=" * 70)
    print("STEP 3: Create train/val/test splits (by identity)")
    print("=" * 70)

    splits = split_by_identity(
        merged,
        train_ratio=config["splits"]["train_ratio"],
        val_ratio=config["splits"]["val_ratio"],
        test_ratio=config["splits"]["test_ratio"],
        seed=config["splits"]["seed"],
    )
    splits_path = os.path.join(config["output"]["manifests_dir"], "dataset_splits.json")
    save_splits(splits, splits_path)

    # ─── Step 4: Compute stats ───
    print("\n" + "=" * 70)
    print("STEP 4: Compute dataset statistics")
    print("=" * 70)

    stats = compute_dataset_stats(merged, splits)
    stats_path = os.path.join(config["output"]["manifests_dir"], "dataset_stats.csv")
    stats.to_csv(stats_path)
    print(f"✅ Stats saved to {stats_path}")
    print(stats.to_string())

    # ─── Step 5: Convert to YOLO format ───
    print("\n" + "=" * 70)
    print("STEP 5: Convert to YOLO detection format")
    print("=" * 70)

    yolo_dir = os.path.join(config["output"]["processed_dir"], "yolo_dataset")
    dataset_yaml = coco_to_yolo(
        coco=merged, splits=splits, output_dir=yolo_dir,
        single_class=True, class_name=args.single_class, symlink=True,
    )
    print(f"✅ YOLO dataset.yaml: {dataset_yaml}")

    # ─── Summary ───
    print("\n" + "=" * 70)
    print("PHASE 0 COMPLETE")
    print("=" * 70)
    print(f"  Merged COCO:    {merged_path}")
    print(f"  Splits:         {splits_path}")
    print(f"  Stats:          {stats_path}")
    print(f"  YOLO dataset:   {dataset_yaml}")
    print(f"\nNext: run scripts/train_detector.py --data {dataset_yaml}")


if __name__ == "__main__":
    main()
