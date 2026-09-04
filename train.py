"""
Construction PPE Detection Model Fine-Tuning Pipeline
================────────────────────────────────======
Adapted from fine-tuning notebook: yolov8-finetuning-for-ppe-detection.ipynb

This script trains/fine-tunes a YOLOv8 model on the Roboflow Construction Site Safety (CSS) dataset
and saves the optimized weights to models/best.pt for risk analysis.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    CLASS_NAMES, DATA_YAML_PATH, MODEL_PATH
)


def ensure_data_yaml(data_dir: Path) -> Path:
    """Create or update data.yaml for YOLO training with correct absolute/relative paths."""
    train_dir = data_dir / "train" / "images"
    val_dir = data_dir / "valid" / "images"
    test_dir = data_dir / "test" / "images"

    data_config = {
        "path": str(data_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    yaml_path = data_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, sort_keys=False)
    
    print(f"[Config] Updated dataset config at: {yaml_path}")
    return yaml_path


def train_model(
    epochs: int = 50,
    batch_size: int = 16,
    base_model: str = "yolov8n.pt",
    img_size: int = 640,
    device: str = "cpu",
    output_dir: str = "runs/detect",
):
    """
    Train or fine-tune YOLOv8 on the Construction PPE Dataset.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size per GPU/CPU iteration.
    base_model : str
        Pretrained YOLO base model ('yolov8n.pt', 'yolov8s.pt', etc.).
    img_size : int
        Input image resolution.
    device : str
        Device for training ('0' for GPU 0, 'cpu' for CPU).
    output_dir : str
        Output directory for training artifacts.
    """
    print("=" * 65)
    print("  FINE-TUNING PIPELINE: Construction PPE Object Detector")
    print("=" * 65)

    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists() or not (data_dir / "train").exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_dir}'. Please ensure data/ directory exists "
            "with train, valid, and test subdirectories."
        )

    yaml_path = ensure_data_yaml(data_dir)

    print(f"\n[1] Initializing base model: {base_model}")
    model = YOLO(base_model)

    print(f"\n[2] Starting training for {epochs} epochs (Batch size: {batch_size}, Image size: {img_size})...")
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=output_dir,
        name="construction_ppe_model",
        exist_ok=True,
        verbose=True,
    )

    # Save best model to models/best.pt
    best_weights = Path(output_dir) / "construction_ppe_model" / "weights" / "best.pt"
    target_weights = Path(MODEL_PATH)
    target_weights.parent.mkdir(parents=True, exist_ok=True)

    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, target_weights)
        print(f"\n[3] ✅ Training complete! Saved best model weights to: {target_weights}")
    else:
        print(f"\n[3] ⚠️ Training finished, but best.pt not found at expected location: {best_weights}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Construction PPE Detector")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt", help="Base model weights (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--img-size", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, 0, etc.)")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_model=args.base_model,
        img_size=args.img_size,
        device=args.device,
    )
