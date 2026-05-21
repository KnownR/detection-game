# 🥋 AI Pose Fighter

A real-time gesture-controlled fighting game built using Computer Vision, Pose Estimation, Deep Learning, and YOLO object detection.

Players perform physical poses in front of a camera to launch attacks and activate shields inside the game.

---

## 🚀 Features

* 🎮 Real-time two-player fighting game
* 🧠 Custom deep learning pose classifier
* 🔥 Fireball attack gesture detection
* 🛡️ Shield gesture recognition
* 👤 Human detection using YOLOv8
* 🦴 Pose estimation using MediaPipe
* ⚡ Optimized real-time gameplay pipeline
* 🎨 Particle effects and visual animations
* 📷 Webcam-based interaction
* 🖥️ Pygame game interface
* 📊 Confidence smoothing and cooldown handling

---

# 📸 Demo

> Add gameplay GIFs/screenshots here.

```text
assets/
├── gameplay.gif
├── screenshot1.png
└── screenshot2.png
```

Example:

```md
![Gameplay](assets/gameplay.gif)
```

---

# 🧠 Technologies Used

| Technology | Purpose                   |
| ---------- | ------------------------- |
| Python     | Core programming language |
| YOLOv8     | Human detection           |
| MediaPipe  | Pose landmark extraction  |
| PyTorch    | Deep learning classifier  |
| OpenCV     | Camera processing         |
| Pygame     | Game rendering            |
| NumPy      | Numerical operations      |

---

# 🏗️ System Architecture

```text
Camera Feed
     ↓
YOLOv8 Human Detection
     ↓
MediaPipe Pose Estimation
     ↓
Custom Pose Classifier
     ↓
Gesture Recognition
     ↓
Game Logic + Attacks + Shields
     ↓
Pygame Rendering
```

---

# 🎯 Supported Actions

| Gesture       | Action                    |
| ------------- | ------------------------- |
| Fireball Pose | Launch attack projectile  |
| Shield Pose   | Activate defensive shield |
| Neutral       | No action                 |

---

# 🧠 Machine Learning Pipeline

The project uses a custom 3-class pose classifier:

* `fireball`
* `shield`
* `neutral`

The classifier is trained using pose landmarks extracted from MediaPipe.

Model file:

```text
pose_3class_resnet.pth
```

Training data:

```text
pose_data_3class.json
```

---

# 📂 Project Structure

```text
.
├── classifier_3class.py          # Pose classification model
├── pose_fighter_3class.py        # Main game engine
├── pose_collector_3class.py      # Dataset collection script
├── train_3class.py               # Training pipeline
├── pose_data_3class.json         # Pose dataset
├── pose_3class_resnet.pth        # Trained model weights
├── yolov8n.pt                    # YOLOv8 detection model
├── yolov8n-seg.pt                # YOLO segmentation model
└── assets/                       # Screenshots / GIFs
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/ai-pose-fighter.git
cd ai-pose-fighter
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Recommended requirements.txt

```txt
opencv-python
numpy
pygame
mediapipe
ultralytics
torch
torchvision
```

---

# ▶️ Run the Game

```bash
python pose_fighter_3class.py
```

---

# 🎮 Gameplay Mechanics

## 👤 Player Detection

Players are detected using:

```python
YOLO("yolov8n.pt")
```

Each player is tracked using bounding boxes and IOU-based matching.

---

## 🦴 Pose Detection

MediaPipe extracts body landmarks in real time.

These landmarks are passed into the custom classifier to determine player actions.

---

## 🔥 Attack System

When the fireball pose is detected:

* Projectile is spawned
* Damage is applied on collision
* Particle effects are generated
* Cooldown logic prevents spamming

---

## 🛡️ Shield System

When the shield pose is detected:

* Temporary shield activates
* Incoming damage is blocked
* Shield expires after cooldown duration

---

# ⚡ Optimizations Implemented

The project includes multiple performance optimizations:

* Confidence smoothing
* Detection cooldowns
* Cached collision rectangles
* Particle pooling concepts
* YOLO confidence filtering
* IOU-based player matching
* Efficient bounding box handling
* Optimized update loops

---

# 📊 Key Concepts Demonstrated

* Computer Vision
* Human Pose Estimation
* Real-time AI inference
* Gesture Recognition
* Deep Learning
* Object Detection
* Interactive Game Systems
* Real-time Rendering
* Event-driven architecture

---

# 🖥️ Hardware Requirements

Recommended:

* GPU-supported system
* Webcam
* Python 3.10+

Optional:

* NVIDIA GPU for faster inference

---

# 🔮 Future Improvements

* Multiplayer networking
* More combat moves
* Voice commands
* Character animations
* TensorRT optimization
* Full-body combo recognition
* Mobile deployment
* Web-based version
* Reinforcement learning opponents

---

# 📸 Suggested Screenshots

Add these for a professional GitHub appearance:

* Gameplay screen
* Pose detection overlay
* Fireball attack moment
* Shield activation
* Model prediction confidence

---

# 📄 License

MIT License

---

# 🙌 Acknowledgements

* YOLOv8 by Ultralytics
* MediaPipe by Google
* PyTorch
* OpenCV
* Pygame

---

# ⭐ If You Like This Project

Consider starring the repository to support the project.
