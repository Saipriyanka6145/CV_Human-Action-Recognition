# AI-Based Human Action & Gesture Recognition Web Application

This directory contains the Streamlit web application layer built around the PYSKL deep learning framework.

## Directory Structure
- `app.py`: Streamlit multi-tab dashboard entry point.
- `inference.py`: Standalone PyTorch ST-GCN++ graph convolutional inference engine.
- `pose_extraction.py`: YOLOv8-Pose (COCO 17-keypoint) and MediaPipe hand tracking pipeline.
- `analytics.py`: Biomechanical motion analytics (velocities, displacements, kinetic energy index).
- `visualization.py`: Plotly interactive charts and OpenCV video HUD annotator.
- `utils.py`: Checkpoint caching, device detection, and label mapping.
- `assets/`: Styling tokens and CSS (`style.css`).

## Quick Start
```bash
python -m streamlit run app/app.py
```
Expected local URL: `http://localhost:8501`
