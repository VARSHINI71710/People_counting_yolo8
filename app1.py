import cv2
from ultralytics import YOLO

# ---------------------------
# 1. Load YOLOv8 model
# ---------------------------
model = YOLO("yolov8n.pt")  # nano model

# ---------------------------
# 2. Video path
# ---------------------------
video_path = "/home/bit-user/Desktop/yolo11peoplecounting/people1.mp4"

# ---------------------------
# 3. Counting parameters
# ---------------------------
line_position = 100
offset = 10
counted_ids = set()
total_count = 0

# Store trajectory points: {track_id: [(x1,y1), (x2,y2), ...]}
trajectories = {}

# ---------------------------
# 4. Track objects
# ---------------------------
tracker = model.track(
    source=video_path,
    stream=True,
    show=False,
    persist=True,
    classes=[0],  # person
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

                track_id = int(box.id[0].item()) if hasattr(box, "id") and box.id is not None else None

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Person’s center point
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Store trajectory points
                if track_id is not None:
                    if track_id not in trajectories:
                        trajectories[track_id] = []
                    trajectories[track_id].append((cx, cy))

                # Draw trajectory path
                if track_id in trajectories and len(trajectories[track_id]) > 1:
                    for i in range(1, len(trajectories[track_id])):
                        cv2.line(frame,
                                 trajectories[track_id][i - 1],
                                 trajectories[track_id][i],
                                 (255, 0, 0), 2)

                # Check crossing line
                if (line_position - offset) < cy < (line_position + offset):
                    if track_id is not None and track_id not in counted_ids:
                        counted_ids.add(track_id)
                        total_count += 1

    # Show count
    cv2.putText(frame, f"Count: {total_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Display video
    cv2.imshow("YOLO People Counter with Trajectory", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):  # slow down video
        break

cv2.destroyAllWindows()
