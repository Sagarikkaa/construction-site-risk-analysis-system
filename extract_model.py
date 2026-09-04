import os
import zipfile
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_PT_PATH = os.path.join(MODELS_DIR, "best.pt")
DOWNLOADS_ZIP = r"C:\Users\Sagarika\Downloads\archive (1).zip"

def extract_best_pt():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.isfile(BEST_PT_PATH):
        print(f"[OK] models/best.pt already exists ({os.path.getsize(BEST_PT_PATH) / (1024*1024):.2f} MB).")
        return True

    if os.path.isfile(DOWNLOADS_ZIP):
        print(f"Extracting best.pt from '{DOWNLOADS_ZIP}'...")
        with zipfile.ZipFile(DOWNLOADS_ZIP, 'r') as z:
            for name in z.namelist():
                if name.endswith("best.pt"):
                    with z.open(name) as src, open(BEST_PT_PATH, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"[OK] Successfully extracted models/best.pt ({os.path.getsize(BEST_PT_PATH) / (1024*1024):.2f} MB).")
                    return True
    
    # Fallback to yolov8n.pt if archive is missing
    print("Downloading fallback YOLOv8n pretrained model...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.save(BEST_PT_PATH)
    print(f"[OK] Created fallback models/best.pt ({os.path.getsize(BEST_PT_PATH) / (1024*1024):.2f} MB).")
    return True

if __name__ == "__main__":
    extract_best_pt()

