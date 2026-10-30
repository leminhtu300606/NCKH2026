"""
Convert COCO JSON to YOLO detection format (for ultralytics training).

Output structure:
  yolo_dir/
    images/
      train/  val/  test/
    labels/
      train/  val/  test/
    dataset.yaml
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm


def coco_to_yolo(
    coco: dict,
    splits: Dict[str, List[int]],
    output_dir: str,
    single_class: bool = True,
    class_name: str = "person",
    symlink: bool = True,
) -> str:
    """
    Convert COCO dict + splits into YOLO directory structure.

    Args:
        coco: COCO annotation dict.
        splits: dict with keys 'train', 'val', 'test' -> list of image_ids.
        output_dir: root of YOLO dataset directory.
        single_class: if True, remap all categories to class 0.
        class_name: name for the single class.
        symlink: if True, create symlinks instead of copying images.

    Returns:
        Path to the generated dataset.yaml.
    """
    output_dir = Path(output_dir)

    # Create directories
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Build lookup
    img_lookup = {img["id"]: img for img in coco["images"]}
    ann_by_image = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # Process each split
    for split_name, image_ids in splits.items():
        if split_name not in ["train", "val", "test"]:
            continue

        for img_id in tqdm(image_ids, desc=f"YOLO {split_name}"):
            if img_id not in img_lookup:
                continue
            img_info = img_lookup[img_id]
            src_path = img_info["file_name"]
            w, h = img_info["width"], img_info["height"]

            # Copy or symlink image
            ext = os.path.splitext(src_path)[1]
            dst_img = output_dir / "images" / split_name / f"{img_id:08d}{ext}"
            if not dst_img.exists():
                if symlink and os.path.isfile(src_path):
                    try:
                        os.symlink(src_path, str(dst_img))
                    except OSError:
                        shutil.copy2(src_path, str(dst_img))
                elif os.path.isfile(src_path):
                    shutil.copy2(src_path, str(dst_img))

            # Write YOLO label
            dst_lbl = output_dir / "labels" / split_name / f"{img_id:08d}.txt"
            anns = ann_by_image.get(img_id, [])
            with open(dst_lbl, "w") as f:
                for ann in anns:
                    bx, by, bw, bh = ann["bbox"]  # COCO absolute
                    # Convert to YOLO normalized (cx, cy, w, h)
                    cx = (bx + bw / 2) / w
                    cy = (by + bh / 2) / h
                    nw = bw / w
                    nh = bh / h
                    # Clamp
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    nw = max(0, min(1, nw))
                    nh = max(0, min(1, nh))

                    cls_id = 0 if single_class else ann["category_id"] - 1
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # Generate dataset.yaml
    names = [class_name] if single_class else [c["name"] for c in coco["categories"]]
    yaml_content = f"""# Auto-generated YOLO dataset config
path: {str(output_dir)}
train: images/train
val: images/val
test: images/test

nc: {len(names)}
names: {names}
"""
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"YOLO dataset written to {output_dir}")
    print(f"  dataset.yaml: {yaml_path}")
    return str(yaml_path)
