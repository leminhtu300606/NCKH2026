"""
Dataset Converter — Phase 0
============================
Discovers the structure of each dataset under its root path,
converts annotations to COCO format, and writes the output JSON.

Supports auto-detection of: YOLO txt, VOC XML, COCO JSON, CSV formats.
"""

import os
import json
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def find_images(root: str) -> List[str]:
    """Recursively find all image files under *root*."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    images: List[str] = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(images)


def find_annotations(root: str) -> Dict[str, List[str]]:
    """Scan for annotation files grouped by extension."""
    result: Dict[str, List[str]] = {"txt": [], "xml": [], "json": [], "csv": []}
    for ext, key in [("*.txt", "txt"), ("*.xml", "xml"), ("*.json", "json"), ("*.csv", "csv")]:
        result[key] = sorted(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return result


def detect_annotation_format(ann_files: Dict[str, List[str]]) -> str:
    """Heuristic to decide which annotation format is present."""
    for jf in ann_files["json"]:
        try:
            with open(jf) as f:
                data = json.load(f)
            if isinstance(data, dict) and "images" in data and "annotations" in data:
                return "coco"
        except Exception:
            pass
    if len(ann_files["txt"]) > 10:
        return "yolo"
    if len(ann_files["xml"]) > 10:
        return "voc"
    if ann_files["csv"]:
        return "csv"
    return "unknown"


# ---------------------------------------------------------------------------
# Format-specific parsers
# ---------------------------------------------------------------------------

def _empty_coco(class_names: List[str]) -> dict:
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]
    return {
        "info": {"description": "Converted by UAV-Privacy pipeline"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }


def _get_or_add_category(coco: dict, name: str) -> int:
    for cat in coco["categories"]:
        if cat["name"] == name:
            return cat["id"]
    new_id = max((c["id"] for c in coco["categories"]), default=0) + 1
    coco["categories"].append({"id": new_id, "name": name})
    return new_id


def parse_yolo_annotations(img_paths: List[str], ann_dir: str,
                           class_names: Optional[List[str]] = None) -> dict:
    coco = _empty_coco(class_names or ["object"])
    ann_id = 1
    for img_id, img_path in enumerate(tqdm(img_paths, desc="Parsing YOLO"), start=1):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        coco["images"].append({"id": img_id, "file_name": img_path, "width": w, "height": h})

        stem = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = None
        for candidate in [os.path.join(ann_dir, stem + ".txt"),
                          os.path.join(os.path.dirname(img_path), stem + ".txt")]:
            if os.path.isfile(candidate):
                txt_path = candidate
                break
        if txt_path is None:
            continue

        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                abs_w, abs_h = bw * w, bh * h
                abs_x, abs_y = cx * w - abs_w / 2, cy * h - abs_h / 2
                bbox = [abs_x, abs_y, abs_w, abs_h]
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": cls_id + 1, "bbox": bbox,
                    "area": abs_w * abs_h, "iscrowd": 0,
                })
                ann_id += 1
    return coco


def parse_voc_annotations(img_paths: List[str], xml_files: List[str],
                          class_names: Optional[List[str]] = None) -> dict:
    coco = _empty_coco(class_names or ["object"])
    ann_id = 1
    xml_map = {os.path.splitext(os.path.basename(xf))[0]: xf for xf in xml_files}

    for img_id, img_path in enumerate(tqdm(img_paths, desc="Parsing VOC"), start=1):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        coco["images"].append({"id": img_id, "file_name": img_path, "width": w, "height": h})

        if stem not in xml_map:
            continue
        tree = ET.parse(xml_map[stem])
        for obj in tree.getroot().findall("object"):
            name = obj.find("name").text
            bb = obj.find("bndbox")
            xmin, ymin = float(bb.find("xmin").text), float(bb.find("ymin").text)
            xmax, ymax = float(bb.find("xmax").text), float(bb.find("ymax").text)
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            cat_id = _get_or_add_category(coco, name)
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id,
                "category_id": cat_id, "bbox": bbox,
                "area": bbox[2] * bbox[3], "iscrowd": 0,
            })
            ann_id += 1
    return coco


def parse_coco_annotations(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def parse_csv_annotations(img_paths: List[str], csv_path: str,
                          class_names: Optional[List[str]] = None) -> dict:
    import pandas as pd
    coco = _empty_coco(class_names or ["object"])
    ann_id = 1
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    img_map = {os.path.basename(ip): ip for ip in img_paths}

    for img_id, (fname, group) in enumerate(tqdm(df.groupby("filename"), desc="Parsing CSV"), start=1):
        img_path = img_map.get(fname)
        if img_path is None:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        coco["images"].append({"id": img_id, "file_name": img_path, "width": w, "height": h})

        for _, row in group.iterrows():
            xmin = float(row.get("xmin", row.get("x_min", 0)))
            ymin = float(row.get("ymin", row.get("y_min", 0)))
            xmax = float(row.get("xmax", row.get("x_max", 0)))
            ymax = float(row.get("ymax", row.get("y_max", 0)))
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            cls_name = str(row.get("class", row.get("label", "object")))
            cat_id = _get_or_add_category(coco, cls_name)
            ann = {"id": ann_id, "image_id": img_id, "category_id": cat_id,
                   "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0}
            attrs = {}
            if "track_id" in row and not pd.isna(row["track_id"]):
                attrs["track_id"] = int(row["track_id"])
            if "identity_id" in row and not pd.isna(row["identity_id"]):
                attrs["identity_id"] = str(row["identity_id"])
            if attrs:
                ann["attributes"] = attrs
            coco["annotations"].append(ann)
            ann_id += 1
    return coco


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert_dataset(root: str, output_path: str, dataset_name: str = "dataset",
                    class_names: Optional[List[str]] = None) -> dict:
    """Auto-discover and convert a dataset under *root* to COCO JSON."""
    print(f"\n{'=' * 60}")
    print(f"Converting dataset: {dataset_name}")
    print(f"Root: {root}")
    print(f"{'=' * 60}")

    images = find_images(root)
    print(f"Found {len(images)} images")

    ann_files = find_annotations(root)
    fmt = detect_annotation_format(ann_files)
    print(f"Detected annotation format: {fmt}")

    if fmt == "coco":
        for jf in ann_files["json"]:
            with open(jf) as f:
                data = json.load(f)
            if isinstance(data, dict) and "images" in data:
                coco = parse_coco_annotations(jf)
                break
    elif fmt == "yolo":
        label_dirs = {os.path.dirname(tf) for tf in ann_files["txt"]}
        ann_dir = sorted(label_dirs)[0] if label_dirs else root
        coco = parse_yolo_annotations(images, ann_dir, class_names)
    elif fmt == "voc":
        coco = parse_voc_annotations(images, ann_files["xml"], class_names)
    elif fmt == "csv":
        coco = parse_csv_annotations(images, ann_files["csv"][0], class_names)
    else:
        print("WARNING: Unknown annotation format. Creating image-only COCO.")
        coco = _empty_coco(class_names or ["object"])
        for img_id, img_path in enumerate(tqdm(images, desc="Indexing images"), start=1):
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            coco["images"].append({"id": img_id, "file_name": img_path, "width": w, "height": h})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Wrote COCO to {output_path}")
    print(f"  Images:      {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Categories:  {[c['name'] for c in coco['categories']]}")
    return coco
