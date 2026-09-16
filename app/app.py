"""
================================================================================
AI-Based Human Action & Gesture Recognition System
Deep Learning Based Human Movement Analysis
================================================================================
A portfolio-grade web application built around the PYSKL graph convolutional
action recognition framework with integrated YOLOv8-Pose and MediaPipe frontend.
"""

import os
import os.path as osp
import sys
import time
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Ensure both project root and app directory are in sys.path
APP_DIR = osp.abspath(osp.dirname(__file__))
ROOT_DIR = osp.abspath(osp.join(APP_DIR, '..'))
for p in [ROOT_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from app.utils import (
        OFFICIAL_CHECKPOINTS,
        get_available_device,
        load_label_map,
        create_temp_dir,
        cleanup_temp_dir
    )
    from app.inference import PYSKLInferencePipeline
    from app.pose_extraction import PoseExtractor, HandGestureExtractor
    from app.analytics import MovementAnalytics
    from app.visualization import (
        create_prediction_chart,
        create_motion_intensity_chart,
        create_body_part_radar,
        render_annotated_video
    )
except (ImportError, ModuleNotFoundError):
    from utils import (
        OFFICIAL_CHECKPOINTS,
        get_available_device,
        load_label_map,
        create_temp_dir,
        cleanup_temp_dir
    )
    from inference import PYSKLInferencePipeline
    from pose_extraction import PoseExtractor, HandGestureExtractor
    from analytics import MovementAnalytics
    from visualization import (
        create_prediction_chart,
        create_motion_intensity_chart,
        create_body_part_radar,
        render_annotated_video
    )


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & THEME
# ==============================================================================

st.set_page_config(
    page_title="AI Human Action & Gesture Recognition",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS Styling
CSS_PATH = osp.join(osp.dirname(__file__), 'assets', 'style.css')
if osp.exists(CSS_PATH):
    with open(CSS_PATH, 'r', encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if 'temp_session_dir' not in st.session_state:
    st.session_state.temp_session_dir = create_temp_dir()

if 'active_model_key' not in st.session_state:
    st.session_state.active_model_key = 'stgcnpp_ntu120'


# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================

with st.sidebar:
    st.markdown("### ⚡ System Configuration")
    
    # Model Selector
    model_options = {
        'stgcnpp_ntu120': "ST-GCN++ (120 Human Actions)",
        'stgcnpp_hagrid': "ST-GCN++ (HaGRID Hand Gestures)"
    }
    selected_model_key = st.selectbox(
        "Select Model Architecture:",
        options=list(model_options.keys()),
        format_func=lambda k: model_options[k],
        index=0 if st.session_state.active_model_key == 'stgcnpp_ntu120' else 1
    )
    st.session_state.active_model_key = selected_model_key

    # Hardware Device Detection & Selector
    default_dev, dev_name = get_available_device()
    device_choice = st.radio(
        "Compute Hardware:",
        options=['cpu', 'cuda:0'] if 'cuda' in default_dev else ['cpu'],
        format_func=lambda x: f"GPU ({dev_name})" if 'cuda' in x else "CPU (Host Processor)"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Parameters")
    
    top_k_val = st.slider("Top-K Candidates:", min_value=1, max_value=10, value=5)
    conf_threshold = st.slider("Confidence Filter Threshold (%):", min_value=0, max_value=80, value=0)
    max_frames_process = st.slider("Max Frames to Process:", min_value=30, max_value=300, value=120, step=15)
    generate_video = st.checkbox("Generate Annotated HUD Video", value=True)

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size: 0.8rem; color: #94a3b8; line-height: 1.4;'>
            <b>Framework:</b> PYSKL GCN Engine<br>
            <b>Pose Extractor:</b> YOLOv8-Pose / MediaPipe<br>
            <b>Status:</b> Ready
        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================================================================
# HERO BANNER & HEADER
# ==============================================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">AI-Based Human Action & Gesture Recognition</div>
        <div class="hero-subtitle">Deep Learning Based Human Movement Analysis & Spatio-Temporal Graph Convolutional Modeling</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# MULTI-PAGE NAVIGATION TABS
# ==============================================================================

tab_overview, tab_analysis, tab_analytics, tab_docs = st.tabs([
    "🏠 Overview & Architecture",
    "🎬 Video Action Analysis",
    "📊 Movement Analytics",
    "📖 Architecture & Attribution"
])


# ==============================================================================
# TAB 1: OVERVIEW & ARCHITECTURE
# ==============================================================================

with tab_overview:
    col_ov1, col_ov2 = st.columns([3, 2])
    
    with col_ov1:
        st.markdown("### 🌟 Project Summary")
        st.markdown(
            """
            This web application provides an **end-to-end computer vision and deep learning system** for recognizing complex human activities and hand gestures from raw video streams. 
            
            By translating raw pixels into structured **spatial-temporal skeleton coordinate graphs**, the application utilizes Graph Convolutional Networks (ST-GCN++) to deliver robust, viewpoint-invariant, and lighting-invariant action classification.
            """
        )

        st.markdown("### 🔄 End-to-End Pipeline Architecture")
        st.markdown(
            """
            <div class="pipeline-step"><span class="pipeline-number">1</span> <b>Input Video Stream:</b> Accepts MP4, AVI, or MOV video clips from camera feeds or file uploads.</div>
            <div class="pipeline-step"><span class="pipeline-number">2</span> <b>Pose & Skeleton Extraction:</b> YOLOv8-Pose tracks COCO 17-keypoint body coordinates across frames.</div>
            <div class="pipeline-step"><span class="pipeline-number">3</span> <b>Spatiotemporal Graph Formation:</b> Normalizes and constructs adjacency matrices representing bone connections.</div>
            <div class="pipeline-step"><span class="pipeline-number">4</span> <b>PYSKL ST-GCN++ Inference:</b> Graph convolution blocks extract spatial topology and multi-scale temporal features.</div>
            <div class="pipeline-step"><span class="pipeline-number">5</span> <b>Softmax Action Prediction:</b> Classifies input into 120 everyday human actions or 40 hand gestures.</div>
            <div class="pipeline-step"><span class="pipeline-number">6</span> <b>Movement Analytics & Visualization:</b> Computes joint velocities, kinetic energy, and renders HUD video.</div>
            """,
            unsafe_allow_html=True
        )

    with col_ov2:
        st.markdown("### 🛠️ Core Technologies")
        tech_df = pd.DataFrame({
            "Component": [
                "Action Recognition Model",
                "Pose Estimation",
                "Deep Learning Runtime",
                "Computer Vision & HUD",
                "Interactive Visualizations",
                "Application Interface"
            ],
            "Technology": [
                "PYSKL (ST-GCN++)",
                "YOLOv8-Pose / MediaPipe",
                "PyTorch 2.x",
                "OpenCV (cv2)",
                "Plotly & Altair",
                "Streamlit Dashboard"
            ]
        })
        st.dataframe(tech_df, hide_index=True, use_container_width=True)

        st.markdown("### 🎯 Supported Capabilities")
        st.info(
            "• **120 Action Classes:** Walking, clapping, falling, reading, hugging, waving, jumping, typing, etc.\n\n"
            "• **40 Gesture Classes:** Peace, Like, Dislike, Fist, OK, Call, Palm, Rock, Mute, Swiping, etc.\n\n"
            "• **Multi-Person Tracking:** Hungarian algorithm assignment across consecutive frames.\n\n"
            "• **Biomechanical Metrics:** Joint displacement curves, velocity profiles, and kinetic motion score."
        )


# ==============================================================================
# TAB 2: VIDEO ACTION ANALYSIS
# ==============================================================================

with tab_analysis:
    st.markdown("### 🎬 Upload Video for Action & Gesture Analysis")
    
    col_input1, col_input2 = st.columns([1, 1])

    with col_input1:
        input_source = st.radio(
            "Select Video Input Source:",
            ["Use Bundled NTU Benchmark Sample (Two-Person Interaction)", "Upload Custom Video (MP4 / AVI / MOV)"],
            horizontal=True
        )

        video_file_path = None

        if "Bundled" in input_source:
            sample_path = osp.join(ROOT_DIR, 'demo', 'ntu_sample.avi')
            if osp.exists(sample_path):
                video_file_path = sample_path
                st.success("✅ Loaded bundled benchmark video: `demo/ntu_sample.avi`")
            else:
                st.error("Sample video not found in `demo/ntu_sample.avi`.")
        else:
            uploaded_file = st.file_uploader(
                "Upload a video clip (max 50MB):",
                type=['mp4', 'avi', 'mov']
            )
            if uploaded_file is not None:
                temp_upload_path = osp.join(st.session_state.temp_session_dir, uploaded_file.name)
                with open(temp_upload_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                video_file_path = temp_upload_path
                st.success(f"✅ Video uploaded: `{uploaded_file.name}` ({round(uploaded_file.size / (1024*1024), 2)} MB)")

    with col_input2:
        if video_file_path and osp.exists(video_file_path):
            st.markdown("#### 📺 Input Video Preview")
            st.video(video_file_path)

    st.markdown("---")
    
    # Run Action Analysis Trigger
    run_col1, run_col2 = st.columns([1, 3])
    with run_col1:
        run_btn = st.button("🚀 Run Action Analysis", type="primary", use_container_width=True)

    if run_btn:
        if not video_file_path or not osp.exists(video_file_path):
            st.warning("⚠️ Please select or upload a video file first.")
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            try:
                start_time = time.time()
                
                # 1. Pose Extraction
                status_text.markdown("⏳ **Step 1/3:** Extracting 2D skeleton joints with YOLOv8-Pose...")
                pose_ext = PoseExtractor(device=device_choice)
                
                def pose_progress(pct, msg):
                    progress_bar.progress(float(pct) * 0.5)
                    status_text.markdown(f"⏳ **Step 1/3:** {msg}")

                pose_data = pose_ext.extract_from_video(
                    video_file_path,
                    max_frames=max_frames_process,
                    progress_callback=pose_progress
                )

                skeletons = pose_data['skeleton_sequence']
                frames = pose_data['frames']
                
                if skeletons.sum() == 0:
                    st.error("❌ No human poses were detected in the provided video. Please try a video with visible subjects.")
                    st.stop()

                # 2. PYSKL GCN Inference
                status_text.markdown(f"⏳ **Step 2/3:** Running {model_options[selected_model_key]} inference...")
                progress_bar.progress(0.6)
                
                pipe = PYSKLInferencePipeline(
                    model_key=selected_model_key,
                    device=device_choice,
                    progress_callback=lambda p, m: status_text.markdown(f"⏳ {m}")
                )
                
                predictions_data = pipe.predict(skeletons, top_k=top_k_val)
                progress_bar.progress(0.8)

                # 3. Motion Analytics
                status_text.markdown("⏳ **Step 3/3:** Computing kinematics and rendering visual outputs...")
                total_proc_time = time.time() - start_time
                analytics = MovementAnalytics(skeletons, fps=pose_data['fps'])
                summary_metrics = analytics.compute_summary_metrics(processing_time_sec=total_proc_time)

                # 4. Video Annotation
                annotated_video_path = None
                if generate_video:
                    out_vid_name = f"annotated_{osp.splitext(osp.basename(video_file_path))[0]}.mp4"
                    annotated_video_path = osp.join(st.session_state.temp_session_dir, out_vid_name)
                    render_annotated_video(
                        frames,
                        skeletons,
                        action_label=predictions_data['top_action'],
                        confidence=predictions_data['top_confidence'],
                        output_path=annotated_video_path,
                        fps=pose_data['fps'],
                        layout='coco' if selected_model_key != 'stgcnpp_hagrid' else 'handmp'
                    )

                progress_bar.progress(1.0)
                status_text.success("✨ Analysis Completed Successfully!")

                # Store in session state
                st.session_state.analysis_results = {
                    'predictions': predictions_data,
                    'summary_metrics': summary_metrics,
                    'annotated_video_path': annotated_video_path,
                    'model_name': model_options[selected_model_key],
                    'video_info': {
                        'total_frames': pose_data['total_frames'],
                        'fps': pose_data['fps'],
                        'duration': pose_data['duration']
                    }
                }

            except Exception as e:
                st.error(f"❌ Error during analysis execution: {str(e)}")
                st.exception(e)

    # Display Results if Available
    if st.session_state.analysis_results is not None:
        res = st.session_state.analysis_results
        preds = res['predictions']
        metrics = res['summary_metrics']

        st.markdown("---")
        st.markdown("## 🎯 Analysis Results & Predictions")

        # Top Prediction Highlight Card
        top_action = preds['top_action']
        top_conf = preds['top_confidence']

        st.markdown(
            f"""
            <div class="prediction-highlight">
                <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; font-weight: 600;">Recognized Human Action</div>
                <div class="pred-label">{top_action}</div>
                <div class="pred-sub">Classification Confidence: <b style="color: #38bdf8;">{top_conf:.2f}%</b> (via {res['model_name']})</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # High-Level Metrics Strip
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">Video Frames</div>
                    <div class="metric-value">{metrics['total_frames']}</div>
                    <div class="metric-badge badge-blue">{metrics['duration_sec']}s total</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">Tracked Persons</div>
                    <div class="metric-value">{metrics['num_tracked_persons']}</div>
                    <div class="metric-badge badge-green">Hungarian Match</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">Processing Time</div>
                    <div class="metric-value">{metrics['processing_time_sec']}s</div>
                    <div class="metric-badge badge-blue">{metrics['processing_fps']} FPS</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">Peak Velocity</div>
                    <div class="metric-value">{int(metrics['peak_joint_velocity_px_s'])}</div>
                    <div class="metric-badge badge-green">px / sec</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m5:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-title">Active Joint</div>
                    <div class="metric-value" style="font-size: 1.25rem; padding-top: 6px;">{metrics['most_active_joint']}</div>
                    <div class="metric-badge badge-blue">Highest Kinetic</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Prediction Chart & Annotated Video
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("#### 📊 Top Action Predictions")
            pred_fig = create_prediction_chart(preds['predictions'])
            st.plotly_chart(pred_fig, use_container_width=True)

            # Table of Top Predictions
            pred_table = pd.DataFrame({
                "Rank": [f"#{i+1}" for i in range(len(preds['predictions']))],
                "Action Class": [p[0].title() for p in preds['predictions']],
                "Confidence Score": [f"{p[1]:.2f}%" for p in preds['predictions']]
            })
            st.dataframe(pred_table, hide_index=True, use_container_width=True)

        with res_col2:
            st.markdown("#### 🎬 Annotated Skeleton Video (HUD)")
            vid_path = res.get('annotated_video_path')
            if vid_path and osp.exists(vid_path):
                st.video(vid_path)
            else:
                st.info("Video annotation was disabled or not generated.")


# ==============================================================================
# TAB 3: MOVEMENT ANALYTICS & BIOMECHANICS
# ==============================================================================

with tab_analytics:
    st.markdown("### 📊 Application-Level Biomechanical & Movement Analytics")
    st.caption("Derived kinematic measurements calculated from 2D skeleton trajectory coordinates.")

    if st.session_state.analysis_results is None:
        st.info("💡 Please run an action analysis in the **Video Action Analysis** tab to view kinematics and movement charts.")
    else:
        res = st.session_state.analysis_results
        metrics = res['summary_metrics']
        timeline = metrics['timeline_motion_intensity']
        fps = res['video_info']['fps']

        col_an1, col_an2 = st.columns([3, 2])

        with col_an1:
            # Motion Intensity Curve
            int_fig = create_motion_intensity_chart(timeline, fps=fps)
            st.plotly_chart(int_fig, use_container_width=True)

        with col_an2:
            # Body Part Radar Chart
            radar_fig = create_body_part_radar(metrics['body_part_activity'])
            st.plotly_chart(radar_fig, use_container_width=True)

        st.markdown("#### 🔬 Detailed Kinematic Profile")
        k_col1, k_col2, k_col3 = st.columns(3)
        with k_col1:
            st.metric("Average Joint Velocity", f"{metrics['avg_joint_velocity_px_s']} px/s")
        with k_col2:
            st.metric("Motion Variation Index", f"{metrics['avg_motion_intensity']} / 100")
        with k_col3:
            st.metric("Kinetic Energy Index", f"{metrics['kinetic_energy_index']}")

        st.markdown("---")
        st.markdown(
            """
            > [!NOTE]
            > **Disclaimer on Analytics:** Derived metrics (velocity, intensity index, displacement) represent image-space 2D pixel measurements from uncalibrated optical cameras and are provided for comparative movement analysis.
            """
        )


# ==============================================================================
# TAB 4: ARCHITECTURE & ATTRIBUTION
# ==============================================================================

with tab_docs:
    st.markdown("### 📖 Architecture, Datasets & Framework Attribution")

    st.markdown(
        """
        #### 🏛️ System Architecture
        This application implements a modular decoupled pipeline separating **Frontend Video Ingestion**, **Feature Extraction (Pose Tracking)**, **Graph Modeling (PYSKL GCN)**, and **Interactive Reporting**:
        """
    )

    st.markdown(
        """
        ```text
        [ Input Video (MP4/AVI) ]
                   │
                   ▼
        [ YOLOv8-Pose Estimator ] ────────► Extracts COCO 17 Keypoints (x, y, conf)
                   │
                   ▼
        [ Inter-Frame Hungarian Tracker ] ─► Resolves Person 0 & Person 1 Tracks
                   │
                   ▼
        [ Spatiotemporal Graph Tensor ] ──► Normalized Tensor: (1, 2, 100, 17, 3)
                   │
                   ▼
        [ PYSKL ST-GCN++ Backbone ] ──────► Spatial Graph Conv + Multi-Scale Temporal Conv
                   │
                   ▼
        [ Classification Head ] ──────────► Softmax Logits across 120 NTU / 40 HaGRID Classes
                   │
                   ▼
        [ Analytics & HUD Generator ] ────► Plotly Charts + Annotated MP4 Stream
        ```
        """
    )

    st.markdown("---")
    st.markdown("#### 📜 Open-Source Framework Attribution")
    st.markdown(
        """
        - **PYSKL (Pose-based Action Recognition):** Developed by the [OpenMMLab / CUHK Multimedia Lab](https://github.com/kennymckormick/pyskl) team. PYSKL provides graph convolutional algorithms including ST-GCN, ST-GCN++, and PoseC3D. Original Apache 2.0 License preserved.
        - **YOLOv8-Pose:** Developed by Ultralytics for real-time pose estimation.
        - **Datasets Used:**
          - **NTU-RGB+D 120:** Large-scale action recognition dataset with 120 human action classes.
          - **HaGRID:** Hand Gesture Recognition Image Dataset with 40 complex hand gesture categories.
        """
    )


# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown(
    """
    <div style="text-align: center; margin-top: 40px; padding: 20px; color: #64748b; font-size: 0.85rem; border-top: 1px solid rgba(255,255,255,0.05);">
        AI-Based Human Action & Gesture Recognition System | Computer Vision Portfolio Project<br>
        Powered by PYSKL & PyTorch | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
