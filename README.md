# AI-Based Human Action & Gesture Recognition System
### *Deep Learning Based Human Movement Analysis & Spatio-Temporal Graph Modeling*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Ultralytics](https://img.shields.io/badge/YOLOv8--Pose-Realtime-00FFFF.svg)](https://ultralytics.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

An end-to-end Computer Vision and Deep Learning web application designed to recognize complex multi-person human activities and hand gestures from video streams. Built around the **PYSKL** graph convolutional framework with an integrated **YOLOv8-Pose** front-end, the system provides real-time skeleton tracking, Spatio-Temporal Graph Convolutions (ST-GCN++), biomechanical movement analytics, and an interactive dashboard.

---

## 📌 Project Overview

Understanding human movement in video is a central challenge in computer vision with critical applications in sports analytics, surveillance, healthcare rehabilitation, and human-computer interaction (HCI). Conventional pixel-based 3D-CNN approaches are computationally heavy and susceptible to lighting variations and background clutter.

This project implements a **skeleton-based action and gesture recognition pipeline**:
1. High-speed 2D keypoint estimation (**YOLOv8-Pose** / **MediaPipe**).
2. Inter-frame person tracking with Hungarian optimal assignment.
3. Spatio-Temporal Graph Convolutional Modeling (**PYSKL ST-GCN++**).
4. Kinematic movement analytics and interactive visual reporting.

---

## ✨ Key Features

- **Dual Recognition Modes:**
  - **Full-Body Actions (120 Classes):** Trained on NTU-RGB+D 120 (e.g., walking, clapping, hugging, waving, jumping, falling, typing).
  - **Hand Gestures (40 Classes):** Trained on the HaGRID dataset (e.g., Peace, Like, Dislike, Fist, OK, Call, Palm, Rock, Mute).
- **Multi-Person Tracking:** Tracks and associates multiple interacting subjects across frames.
- **Biomechanical Movement Analytics:** Computes real-time joint displacements, velocity profiles, Kinetic Motion Index, and body quadrant motion contribution.
- **Video HUD Annotator:** Renders processed videos with animated skeleton overlays and real-time HUD prediction banners.
- **Portfolio-Grade Web Interface:** Built with Streamlit, custom dark glassmorphism styling, Plotly charts, and hardware device selector (CPU/GPU).
- **Zero Complex Toolchain Friction:** Pure PyTorch runtime that runs out-of-the-box on modern Python and Windows/Linux environments without requiring legacy C++ compilation.

---

## 🏛️ System Architecture

```text
                               ┌────────────────────────────────┐
                               │       Input Video Stream       │
                               │        (MP4, AVI, MOV)         │
                               └────────────────┬───────────────┘
                                                │
                                                ▼
                               ┌────────────────────────────────┐
                               │   YOLOv8-Pose / MediaPipe      │
                               │   Front-End Pose Extractor     │
                               └────────────────┬───────────────┘
                                                │
                                                ▼
                               ┌────────────────────────────────┐
                               │   Hungarian Skeleton Tracker   │
                               │    (Person 0 & Person 1 Tracks)│
                               └────────────────┬───────────────┘
                                                │
                                                ▼
                               ┌────────────────────────────────┐
                               │   Spatio-Temporal Graph Tensor │
                               │       (1, M, T, V, C)          │
                               └────────────────┬───────────────┘
                                                │
                                                ▼
                               ┌────────────────────────────────┐
                               │    PYSKL ST-GCN++ Backbone     │
                               │  Spatial Graph + MS-TCN Convs  │
                               └────────────────┬───────────────┘
                                                │
                                                ▼
                               ┌────────────────────────────────┐
                               │      Classification Head       │
                               │     (Softmax Probability)      │
                               └────────────────┬───────────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
    ┌─────────────────────────────────┐                   ┌─────────────────────────────────┐
    │     Top-K Action Predictions    │                   │   Movement Kinematics & HUD     │
    │  Confidence Distribution Chart  │                   │   Annotated Video Playback      │
    └─────────────────────────────────┘                   └─────────────────────────────────┘
```

---

## 🔬 How Pose Extraction & PYSKL Work Together

It is critical to distinguish the role of each component in this architecture:

1. **Pose Estimation (Front-End Feature Extractor):**
   - **PYSKL is not a pose detector**; it is an action recognition framework that requires skeleton coordinates as input.
   - Our application integrates **YOLOv8-Pose** to extract 17 COCO body keypoints *(Nose, Eyes, Ears, Shoulders, Elbows, Wrists, Hips, Knees, Ankles)* or **MediaPipe** for 21 hand landmarks across every video frame.
   - An inter-frame Hungarian matching algorithm tracks individual persons over time, constructing an $(M, T, V, C)$ tensor.

2. **Action Recognition (PYSKL ST-GCN++ Core):**
   - The skeleton sequence is normalized and fed into the **PYSKL ST-GCN++** model.
   - **Spatial Graph Convolutions** model physical bone connectivity and body topology.
   - **Multi-Scale Temporal Convolutions (MS-TCN)** capture fast and slow dynamics across time branches.
   - The classification head produces softmax probability distributions across candidate action classes.

---

## 📊 Biomechanical Movement Analytics

The application derives real-time kinematic parameters from image-space skeleton trajectories:
- **Instantaneous Joint Velocity ($v = \frac{\Delta d}{\Delta t}$):** Frame-to-frame pixel displacement divided by frame duration.
- **Motion Intensity Index (0–100):** Normalized aggregate acceleration and movement magnitude across all keypoints.
- **Kinetic Energy Proxy ($E_k \propto \sum v_i^2$):** Relative energy expenditure indicator.
- **Body Quadrant Contribution:** Percentage of total movement generated by the Upper Body (arms/shoulders), Core/Torso, Lower Body (legs), and Head/Neck.

---

## 📂 Project Structure

```text
pyskl/
├── app/                               # Portfolio Application Layer
│   ├── app.py                         # Streamlit multi-tab dashboard entrypoint
│   ├── inference.py                   # PYSKL ST-GCN++ PyTorch inference engine
│   ├── pose_extraction.py             # YOLOv8-Pose and MediaPipe keypoint extractor
│   ├── analytics.py                   # Kinematic and movement analytics engine
│   ├── visualization.py               # Plotly interactive charts & OpenCV video annotator
│   ├── utils.py                       # Checkpoint caching, device detection, label map
│   └── assets/
│       └── style.css                  # Custom dark glassmorphism styling tokens
│
├── configs/                           # PYSKL model configurations
├── demo/                              # Sample data & checkpoints
│   ├── hagrid.pth                     # Pretrained HaGRID gesture checkpoint
│   └── ntu_sample.avi                 # Sample benchmark video clip
├── tools/                             # PYSKL data preparation & label maps
│   └── data/label_map/nturgbd_120.txt # 120 action class labels
├── requirements.txt                   # Application dependencies
├── .gitignore                         # Excludes cache, large weights, and temporary files
└── README.md                          # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone & Navigate to Repository
```bash
cd "D:\ML Projects\pyskl"
```

### 2. Install Dependencies
Ensure you have Python (>= 3.10) installed. Install required packages:
```bash
pip install -r requirements.txt
```

### 3. Pretrained Model Weights
- **HaGRID Gesture Model:** Already included locally at `demo/hagrid.pth` (927 KB).
- **NTU-120 Action Recognition Model:** Automatically downloaded on first use from the official OpenMMLab PYSKL release (`stgcnpp_ntu120_hrnet_j.pth`, ~5.9 MB) and cached in `checkpoints/`.

---

## 🖥️ Running the Web Application

Launch the Streamlit web application:

```bash
python -m streamlit run app/app.py
```

Open your browser and navigate to:
```text
http://localhost:8501
```

---

## 💡 Usage Workflow

1. **Select Model Architecture:** Choose between *ST-GCN++ (120 Human Actions)* or *ST-GCN++ (HaGRID Gestures)* in the sidebar.
2. **Choose Video Source:**
   - Click **Use Bundled NTU Benchmark Sample** for an instant demonstration with `demo/ntu_sample.avi`.
   - Or upload any custom `MP4`, `AVI`, or `MOV` video file.
3. **Configure Settings:** Adjust Top-K candidates, confidence threshold, and maximum frame limits.
4. **Click "Run Action Analysis":**
   - Step 1: YOLOv8-Pose tracks human skeletons across frames.
   - Step 2: PYSKL ST-GCN++ computes action classification.
   - Step 3: Analytics and HUD annotated video are generated.
5. **Explore Results:** Inspect the top predicted action, confidence distributions, interactive Plotly charts, kinematic metrics, and annotated video playback.

---

## ⚠️ Limitations & Technical Scope

- **Camera Calibration:** Motion analytics (velocities, displacements) are calculated in 2D image pixel space and represent uncalibrated comparative measurements.
- **Extreme Occlusions:** If subjects are heavily occluded or out of frame, pose extraction confidence may drop.

---

## 📜 Attribution & Open Source Acknowledgement

- **PYSKL Framework:** Developed by [Haodong Duan, Jiaqi Wang, et al.](https://github.com/kennymckormick/pyskl) under the Apache 2.0 License.
- **YOLOv8:** Developed by [Ultralytics](https://github.com/ultralytics/ultralytics).
- **NTU-RGB+D:** Dataset provided by ROSE Lab, Nanyang Technological University.

---

## 📄 License

This project is licensed under the **Apache 2.0 License** - see the [LICENSE](LICENSE) file for details.