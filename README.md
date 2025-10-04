# 🚶‍♂️ YOLOv8 People Counter with Trajectory Tracking 📹

A Python-based project that uses **YOLOv8 object detection and tracking** to count people crossing a line in a video and visualize their movement trajectories.

This project detects people in a video, assigns unique IDs, tracks their movement, and draws **trajectory paths** for each person while counting crossings.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🕒 Real-time detection | Detect people in videos with YOLOv8        |
| 🆔 Unique ID tracking  | Assigns unique IDs to tracked people       |
| 🚦 Counting line       | Counts people crossing a user-defined line |
| 📈 Trajectory tracking | Shows colored movement trajectories        |
| ⚙ Adjustable line      | Change line position for counting          |
| 📺 Live display        | Shows bounding boxes, IDs, count & trajectories |

---

## 📂 Project Structure

YOLO-People-Counter/
│
├── app.py # Main Python script

├── requirements.txt # Python dependencies

├── README.md # Project description

└── people.mp4 # Sample video

## 🛠 Installation

1️⃣ Clone this repository:
```bash
git clone https://github.com/yourusername/YOLO-People-Counter.git
cd YOLO-People-Counter

2️⃣ Install dependencies:
bash
Copy code
pip install ultralytics opencv-python

3️⃣ (Optional) Upgrade YOLO:
bash
Copy code
pip install --upgrade ultralytics

⚙ Usage

Step	Command / Code
1	Place your video in the project folder and set the path in app.py:
python video_path = "/path/to/your/video.mp4"

2	Adjust counting line position in app.py:
python line_position = 600

3	Run the script:
bash python app.py



🧠 How It Works
Loads a pre-trained YOLOv8 model (yolov8n.pt).

Processes the video frame-by-frame.

For each detected person:

Draws bounding box and ID.

Stores center positions.

Draws a colored trajectory line for movement.

Checks if person crossed the counting line.

Displays live count on the video.

📌 Requirements

Python >= 3.8
Ultralytics YOLO
OpenCV (opencv-python)

Install dependencies:
bash
copy code
pip install ultralytics opencv-python

🎥 Example Output

The live video output will display:

✅ Bounding boxes around detected people

🆔 Unique IDs for each person

📈 Colored trajectory lines showing movement

🚦 Counting line and total count displayed

