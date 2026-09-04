"""Raw YOLO diagnostic for the uploaded construction video."""
import os
import cv2
from ultralytics import YOLO
from utils.config import MODEL_PATH

VIDEO_PATH = r"C:\Users\Sagarika\Videos\Screen Recordings\Screen Recording 2026-09-02 192725.mp4"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "video_debug")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
model_names = {int(class_id): str(name) for class_id, name in model.names.items()}
print(f"MODEL PATH: {MODEL_PATH}")
print(f"MODEL TYPE: {type(model).__name__}, task={model.task}")
print(f"NUMBER OF CLASSES: {len(model_names)}")
print("CLASS NAMES:")
for class_id, name in model_names.items():
    print(f"{class_id}: {name}")

capture = cv2.VideoCapture(VIDEO_PATH)
if not capture.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

for frame_number in range(1, 11):
    success, frame = capture.read()
    if not success:
        print(f"FRAME {frame_number}\nCould not read frame")
        break

    results = model.predict(frame, conf=0.25, verbose=False)
    result = results[0]
    boxes = result.boxes
    print(f"\nFRAME {frame_number}")
    print(f"Detection count: {0 if boxes is None else len(boxes)}")
    print("class_id | class_name | confidence | bbox")
    print("------------------------------------------------")

    annotated = frame.copy()
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model_names.get(class_id, f"class_{class_id}")
            confidence = float(box.conf[0])
            bbox = [round(value, 1) for value in box.xyxy[0].tolist()]
            print(f"{class_id} | {class_name} | {confidence:.3f} | {bbox}")
            x1, y1, x2, y2 = [int(value) for value in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, f"{class_name} {confidence:.2f}", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    if frame_number <= 3:
        output_path = os.path.join(OUTPUT_DIR, f"sample_{frame_number}.jpg")
        cv2.imwrite(output_path, annotated)
        print(f"Saved: {output_path}")

capture.release()

print("\nDASHBOARD SAMPLING (frame_skip=10, max_frames=30)")
capture = cv2.VideoCapture(VIDEO_PATH)
sample_index = 0
frame_index = 0
while capture.isOpened() and sample_index < 30:
    success, frame = capture.read()
    if not success:
        break
    if frame_index % 10 == 0:
        result = model.predict(frame, conf=0.25, verbose=False)[0]
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                detections.append((
                    class_id,
                    model_names.get(class_id, f"class_{class_id}"),
                    float(box.conf[0]),
                ))
        print(f"Sample {sample_index + 1}: frame {frame_index}: {detections or 'no detections'}")
        if sample_index < 3:
            annotated = result.plot()
            output_path = os.path.join(OUTPUT_DIR, f"sample_{sample_index + 1}.jpg")
            cv2.imwrite(output_path, annotated)
        sample_index += 1
    frame_index += 1
capture.release()
