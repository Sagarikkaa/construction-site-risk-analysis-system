"""
Convert Ultralytics NDJSON Dataset to YOLO Format
=================================================
Downloads images and generates YOLO annotation files (.txt) and data.yaml
from an Ultralytics HUB / Platform NDJSON export.
"""

import os
import sys
import json
import time
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Project 10 classes
PROJECT_CLASS_NAMES = [
    "Hardhat",        # 0
    "Mask",           # 1
    "NO-Hardhat",     # 2
    "NO-Mask",        # 3
    "NO-Safety Vest", # 4
    "Person",         # 5
    "Safety Cone",    # 6
    "Safety Vest",    # 7
    "machinery",      # 8
    "vehicle",        # 9
]

# NDJSON original 11 classes:
# 0: helmet, 1: gloves, 2: vest, 3: boots, 4: goggles, 5: none,
# 6: Person, 7: no_helmet, 8: no_goggle, 9: no_gloves, 10: no_boots
NDJSON_TO_PROJECT_MAP = {
    0: 0,  # helmet -> Hardhat
    7: 2,  # no_helmet -> NO-Hardhat
    5: 4,  # none (torso without vest) -> NO-Safety Vest
    6: 5,  # Person -> Person
    2: 7,  # vest -> Safety Vest
}

NDJSON_FULL_CLASSES = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "none",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
]


def download_single_image(url: str, dest_path: Path, max_retries: int = 3, timeout: int = 15) -> bool:
    """Download image with retry mechanism."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read()
                if len(content) > 0:
                    with open(dest_path, "wb") as f:
                        f.write(content)
                    return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[Error] Failed to download {dest_path.name} from {url[:60]}... : {e}")
            time.sleep(0.5 * (attempt + 1))
    return False


def convert_dataset(
    ndjson_path: str,
    output_dir: str,
    num_workers: int = 16,
):
    ndjson_file = Path(ndjson_path).resolve()
    target_dir = Path(output_dir).resolve()
    
    if not ndjson_file.exists():
        raise FileNotFoundError(f"NDJSON file not found at: {ndjson_file}")

    print("=" * 70)
    print(f"  CONVERTING ULTRALYTICS NDJSON TO YOLO DATASET")
    print(f"  Source: {ndjson_file}")
    print(f"  Output: {target_dir}")
    print("=" * 70)

    # 1. Parse NDJSON
    records = []
    dataset_metadata = {}
    with open(ndjson_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("type") == "dataset":
                dataset_metadata = item
            elif item.get("type") == "image":
                records.append(item)

    total_images = len(records)
    print(f"[1/4] Parsed {total_images} image records from NDJSON.")
    print(f"      Dataset name: {dataset_metadata.get('name', 'N/A')}")

    # 2. Prepare directories and label files
    # Splits: map 'val' -> 'valid' for consistency with Roboflow
    split_map = {"train": "train", "val": "valid", "test": "test"}

    download_tasks = []
    stats_boxes_compatible = 0
    stats_boxes_full = 0

    print(f"[2/4] Generating YOLO annotation text files...")
    for item in records:
        filename = item["file"]
        raw_split = item.get("split", "train")
        split = split_map.get(raw_split, raw_split)
        url = item["url"]
        boxes = item.get("annotations", {}).get("boxes", [])

        img_dest = target_dir / split / "images" / filename
        txt_compat_dest = target_dir / split / "labels" / f"{Path(filename).stem}.txt"
        txt_full_dest = target_dir / split / "labels_full" / f"{Path(filename).stem}.txt"

        txt_compat_dest.parent.mkdir(parents=True, exist_ok=True)
        txt_full_dest.parent.mkdir(parents=True, exist_ok=True)

        # Build compatible labels
        compat_lines = []
        full_lines = []
        for box in boxes:
            cls_id = int(box[0])
            xc, yc, w, h = box[1], box[2], box[3], box[4]
            
            # Full 11-class line
            full_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            stats_boxes_full += 1

            # Compatible 10-class line
            if cls_id in NDJSON_TO_PROJECT_MAP:
                mapped_cls = NDJSON_TO_PROJECT_MAP[cls_id]
                compat_lines.append(f"{mapped_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                stats_boxes_compatible += 1

        with open(txt_compat_dest, "w", encoding="utf-8") as f:
            f.write("\n".join(compat_lines) + ("\n" if compat_lines else ""))

        with open(txt_full_dest, "w", encoding="utf-8") as f:
            f.write("\n".join(full_lines) + ("\n" if full_lines else ""))

        download_tasks.append((url, img_dest))

    print(f"      Created compatible label files: {stats_boxes_compatible} object instances mapped.")
    print(f"      Created full benchmark label files: {stats_boxes_full} object instances.")

    # 3. Create YAML config files
    print(f"[3/4] Writing dataset YAML configurations...")
    data_yaml_compat = {
        "path": str(target_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(PROJECT_CLASS_NAMES),
        "names": PROJECT_CLASS_NAMES,
    }
    with open(target_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_compat, f, sort_keys=False)

    data_yaml_full = {
        "path": str(target_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(NDJSON_FULL_CLASSES),
        "names": NDJSON_FULL_CLASSES,
    }
    with open(target_dir / "data_full.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_full, f, sort_keys=False)

    # 4. Multi-threaded Download
    print(f"[4/4] Starting concurrent download of {len(download_tasks)} images ({num_workers} worker threads)...")
    start_time = time.time()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(download_single_image, url, dest): dest
            for url, dest in download_tasks
        }
        for future in as_completed(future_to_task):
            dest = future_to_task[future]
            try:
                success = future.result()
                if success:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"[Exception] {dest.name}: {e}")
                failed += 1

            if (completed + failed) % 150 == 0 or (completed + failed) == total_images:
                elapsed = time.time() - start_time
                rate = (completed + failed) / (elapsed + 1e-5)
                print(f"      Progress: {completed + failed}/{total_images} ({completed} succeeded, {failed} failed) - {rate:.1f} img/s")

    elapsed_total = time.time() - start_time
    print("=" * 70)
    print(f"  DOWNLOAD & CONVERSION COMPLETED in {elapsed_total:.1f}s")
    print(f"  Success: {completed} images | Failed: {failed} images")
    print(f"  Dataset directory: {target_dir}")
    print("=" * 70)
    return completed, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Ultralytics NDJSON to YOLO format")
    parser.add_argument(
        "--ndjson",
        type=str,
        default=r"C:\Users\Sagarika\Downloads\construction-ppe.ndjson",
        help="Path to .ndjson file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "ultralytics_ppe"),
        help="Output directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of download threads",
    )
    args = parser.parse_args()
    convert_dataset(args.ndjson, args.output, args.workers)
