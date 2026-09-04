"""
Diagnostic script — inspect the actual YOLO model's class names
and compare them against what config.py expects.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from utils.config import MODEL_PATH, CLASS_NAMES

print("=" * 70)
print("  DIAGNOSTIC: YOLO Model Class Name Inspection")
print("=" * 70)

# 1. Load the model
print(f"\n[1] Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# 2. Print the model's actual class names
actual_names = model.names  # dict: {0: 'name', 1: 'name', ...}
print(f"\n[2] ACTUAL model class names (model.names):")
print(f"    Type: {type(actual_names)}")
print(f"    Number of classes: {len(actual_names)}")
for class_id, class_name in sorted(actual_names.items()):
    print(f"    {class_id}: '{class_name}'")

# 3. Print what config.py expects
print(f"\n[3] CONFIG CLASS_NAMES (from utils/config.py):")
print(f"    Number of classes: {len(CLASS_NAMES)}")
for i, name in enumerate(CLASS_NAMES):
    print(f"    {i}: '{name}'")

# 4. Compare
print(f"\n[4] COMPARISON:")
mismatch = False
for class_id in sorted(actual_names.keys()):
    actual = actual_names[class_id]
    expected = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "N/A"
    match = "✓" if actual == expected else "✗ MISMATCH"
    if actual != expected:
        mismatch = True
    print(f"    ID {class_id}: model='{actual}' | config='{expected}'  {match}")

if not mismatch:
    print("\n    ✅ All class names MATCH between model and config.")
else:
    print("\n    ❌ MISMATCH detected! The config CLASS_NAMES do NOT match the model.")

# 5. Check what the model actually returns for each semantic group
from utils.config import HAZARD_CLASSES, WORKER_CLASSES, SAFETY_EQUIPMENT_CLASSES, HEAVY_OBJECTS

print(f"\n[5] SEMANTIC GROUP CHECK:")
actual_set = set(actual_names.values())

print(f"\n  WORKER_CLASSES expected: {WORKER_CLASSES}")
for w in WORKER_CLASSES:
    status = "✓ EXISTS" if w in actual_set else "✗ NOT IN MODEL"
    print(f"    '{w}': {status}")

print(f"\n  HAZARD_CLASSES expected: {HAZARD_CLASSES}")
for h in HAZARD_CLASSES:
    status = "✓ EXISTS" if h in actual_set else "✗ NOT IN MODEL"
    print(f"    '{h}': {status}")

print(f"\n  SAFETY_EQUIPMENT_CLASSES expected: {SAFETY_EQUIPMENT_CLASSES}")
for s in SAFETY_EQUIPMENT_CLASSES:
    status = "✓ EXISTS" if s in actual_set else "✗ NOT IN MODEL"
    print(f"    '{s}': {status}")

print(f"\n  HEAVY_OBJECTS expected: {HEAVY_OBJECTS}")
for obj in HEAVY_OBJECTS:
    status = "✓ EXISTS" if obj in actual_set else "✗ NOT IN MODEL"
    print(f"    '{obj}': {status}")

# 6. Check model task type
print(f"\n[6] MODEL DETAILS:")
print(f"    Model task: {getattr(model, 'task', 'unknown')}")
print(f"    Model type: {type(model).__name__}")

print("\n" + "=" * 70)
print("  DIAGNOSTIC COMPLETE")
print("=" * 70)
