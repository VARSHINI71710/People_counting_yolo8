import cv2
from ultralytics import YOLO

# ---------------------------
# 1. Load YOLOv11 model
# ---------------------------
# This will auto-download yolo11n.pt if missing
model = YOLO("yolov8n.pt")  # nano model
  # No need to manually download weights

# ---------------------------
# 2. Video path
# ---------------------------
video_path = "//home/bit-user/Desktop/yolo11peoplecounting/people.mp4"

# ---------------------------
# 3. Counting parameters
# ---------------------------
line_position = 600 # y-coordinate of counting line
offset = 10          # tolerance for crossing
counted_ids = set()  # to store unique tracked IDs
total_count = 0

# ---------------------------
# 4. Track objects
# ---------------------------
tracker = model.track(
    source=video_path,
    stream=True,
    show=False,
    persist=True,
    classes=[0],  # class 0 = person in COCO dataset
)

# ---------------------------
# 5. Process frames
# ---------------------------
for result in tracker:
    frame = result.orig_img.copy()

    # Draw counting line
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 255), 3)

    # Loop through detections
    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls[0].item()) if hasattr(box.cls, "__getitem__") else int(box.cls.item())
            if cls == 0:  # person
                coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, "__getitem__") else box.xyxy.cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)

                # Get tracking ID
                track_id = int(box.id[0].item()) if hasattr(box, "id") and box.id is not None else None

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Draw center point
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Check crossing line
                if (line_position - offset) < cy < (line_position + offset):
                    if track_id is not None and track_id not in counted_ids:
                        counted_ids.add(track_id)
                        total_count += 1

    # Show count
    cv2.putText(frame, f"Count: {total_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Display video
    cv2.imshow("YOLOv11 People Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
