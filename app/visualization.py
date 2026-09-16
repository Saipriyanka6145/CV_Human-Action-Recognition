"""
Visualization Suite
Interactive Plotly Charts (Top-K Predictions, Motion Intensity, Body Activity Radar)
and OpenCV Skeleton Overlay Video Renderer with HUD.
"""

import os
import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import sys
import os.path as osp

APP_DIR = osp.abspath(osp.dirname(__file__))
ROOT_DIR = osp.abspath(osp.join(APP_DIR, '..'))
for p in [ROOT_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from app.pose_extraction import COCO_SKELETON_EDGES, HAND_SKELETON_EDGES
    from app.analytics import JOINT_NAMES_COCO
except (ImportError, ModuleNotFoundError):
    from pose_extraction import COCO_SKELETON_EDGES, HAND_SKELETON_EDGES
    from analytics import JOINT_NAMES_COCO


# Color Palette for Person Skeletons (BGR format for OpenCV)
PERSON_COLORS = [
    {
        'bone': (0, 230, 115),      # Neon Green
        'joint': (0, 255, 255),     # Yellow
        'head': (255, 105, 180)     # Pink
    },
    {
        'bone': (255, 140, 0),      # Neon Blue/Cyan
        'joint': (255, 215, 0),     # Gold
        'head': (147, 20, 255)      # Deep Pink
    }
]


# ==============================================================================
# PLOTLY INTERACTIVE CHARTS
# ==============================================================================

def create_prediction_chart(predictions):
    """
    Generate horizontal bar chart for Top-K predicted actions with confidence scores.
    Args:
        predictions: list of (action_name, confidence_percent)
    """
    actions = [p[0].title() for p in reversed(predictions)]
    scores = [p[1] for p in reversed(predictions)]
    colors = ['#10b981' if i == len(predictions) - 1 else '#3b82f6' for i in range(len(predictions))]

    fig = go.Figure(go.Bar(
        x=scores,
        y=actions,
        orientation='h',
        marker=dict(
            color=scores,
            colorscale=[[0, '#1e293b'], [0.5, '#3b82f6'], [1.0, '#10b981']],
            line=dict(color='#38bdf8', width=1.5)
        ),
        text=[f"{s:.1f}%" for s in scores],
        textposition='outside',
        textfont=dict(color='#ffffff', size=13, family='Inter, sans-serif')
    ))

    fig.update_layout(
        title=dict(
            text="🎯 Action Prediction Confidence Distribution",
            font=dict(size=16, color="#f8fafc", family="Inter, sans-serif")
        ),
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        xaxis=dict(
            title=dict(text="Confidence (%)", font=dict(color="#94a3b8")),
            range=[0, max(100, max(scores) * 1.15)],
            gridcolor='rgba(255, 255, 255, 0.08)',
            tickfont=dict(color="#cbd5e1")
        ),
        yaxis=dict(
            tickfont=dict(color="#f1f5f9", size=13),
            gridcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=20, r=40, t=50, b=30),
        height=320
    )
    return fig


def create_motion_intensity_chart(intensity_timeline, fps=25.0):
    """
    Generate line chart showing motion intensity variation across video timeline.
    """
    times = [i / fps for i in range(len(intensity_timeline))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=intensity_timeline,
        mode='lines',
        line=dict(color='#38bdf8', width=2.5, shape='spline'),
        fill='tozeroy',
        fillcolor='rgba(56, 189, 248, 0.15)',
        name='Motion Intensity'
    ))

    fig.update_layout(
        title=dict(
            text="⚡ Temporal Motion Intensity Profile",
            font=dict(size=16, color="#f8fafc", family="Inter, sans-serif")
        ),
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        xaxis=dict(
            title=dict(text="Timestamp (seconds)", font=dict(color="#94a3b8")),
            gridcolor='rgba(255, 255, 255, 0.08)',
            tickfont=dict(color="#cbd5e1")
        ),
        yaxis=dict(
            title=dict(text="Intensity Index (0-100)", font=dict(color="#94a3b8")),
            range=[0, 105],
            gridcolor='rgba(255, 255, 255, 0.08)',
            tickfont=dict(color="#cbd5e1")
        ),
        margin=dict(l=20, r=20, t=50, b=30),
        height=300
    )
    return fig


def create_body_part_radar(activity_breakdown):
    """
    Generate polar/radar chart for body quadrant motion contribution.
    """
    categories = list(activity_breakdown.keys())
    values = list(activity_breakdown.values())
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(168, 85, 247, 0.25)',
        line=dict(color='#a855f7', width=2),
        marker=dict(size=6, color='#c084fc'),
        name='Motion Share'
    ))

    fig.update_layout(
        title=dict(
            text="🧘 Body Quadrant Engagement",
            font=dict(size=16, color="#f8fafc", family="Inter, sans-serif")
        ),
        polar=dict(
            bgcolor='rgba(15, 23, 42, 0.6)',
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2 if values else 100],
                gridcolor='rgba(255, 255, 255, 0.1)',
                tickfont=dict(color="#94a3b8", size=10)
            ),
            angularaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)',
                tickfont=dict(color="#e2e8f0", size=12)
            )
        ),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=40, r=40, t=50, b=40),
        height=320
    )
    return fig


# ==============================================================================
# OPENCV VIDEO HUD RENDERER
# ==============================================================================

def render_annotated_video(frames, skeleton_sequence, action_label, confidence, output_path, fps=25.0, layout='coco'):
    """
    Render skeleton overlay and top HUD banner on video frames and write to output file.
    Args:
        frames: list of RGB or BGR frames
        skeleton_sequence: np.ndarray of shape (M, T, V, 2 or 3)
        action_label: str
        confidence: float (0-100)
        output_path: str destination file path
        fps: float
        layout: 'coco' or 'handmp'
    """
    if len(frames) == 0:
        return output_path

    h, w = frames[0].shape[:2]
    edges = HAND_SKELETON_EDGES if layout == 'handmp' else COCO_SKELETON_EDGES
    num_persons = skeleton_sequence.shape[0]
    total_frames = len(frames)

    # Use OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for f_idx in range(total_frames):
        frame = frames[f_idx].copy()

        # Draw Skeletons
        for m in range(num_persons):
            kpts = skeleton_sequence[m, f_idx] if f_idx < skeleton_sequence.shape[1] else None
            if kpts is None or np.all(kpts == 0):
                continue

            color_cfg = PERSON_COLORS[m % len(PERSON_COLORS)]

            # Draw bones
            for u, v in edges:
                if u < len(kpts) and v < len(kpts):
                    pt1 = (int(kpts[u, 0]), int(kpts[u, 1]))
                    pt2 = (int(kpts[v, 0]), int(kpts[v, 1]))
                    # Validate points
                    if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                        cv2.line(frame, pt1, pt2, color_cfg['bone'], 3, cv2.LINE_AA)

            # Draw joints
            for j_idx, pt in enumerate(kpts):
                x, y = int(pt[0]), int(pt[1])
                if x > 0 and y > 0:
                    cv2.circle(frame, (x, y), 5, color_cfg['joint'], -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), 6, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw Top HUD Banner
        hud_height = 65
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_height), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # HUD Text
        hud_title = f"ACTION: {action_label.upper()}"
        hud_conf = f"CONFIDENCE: {confidence:.1f}%"
        hud_frame = f"FRAME: {f_idx + 1}/{total_frames}"

        cv2.putText(frame, hud_title, (16, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 180), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{hud_conf}  |  {hud_frame}", (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 230, 240), 1, cv2.LINE_AA)

        # Bottom Subtitle Pill
        cv2.rectangle(frame, (0, hud_height), (w, hud_height + 2), (56, 189, 248), -1)

        writer.write(frame)

    writer.release()
    return output_path
