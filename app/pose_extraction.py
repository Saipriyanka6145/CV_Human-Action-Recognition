"""
Pose and Skeleton Extraction Module
Uses YOLOv8-Pose for COCO 17-Keypoint Body Pose Tracking and
Landmark extraction for Hand Gestures.
"""

import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

COCO_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Upper body & arms
    (5, 11), (6, 12), (11, 12),              # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)   # Lower body & legs
]

HAND_SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (0, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)    # Pinky
]


def dist_ske(ske1, ske2):
    """Compute distance between two skeleton instances for tracking assignment."""
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2]) if ske1.shape[1] > 2 and ske2.shape[1] > 2 else 0
    return np.sum(np.maximum(dist, diff))


def track_poses(frame_poses_list, max_tracks=2, max_frame_gap=15):
    """
    Formulate frame-wise pose detections into coherent multi-person tracks.
    Args:
        frame_poses_list: list of length T, each element is a list of poses [(17, 3), ...]
    Returns:
        track_array: np.ndarray of shape (max_tracks, T, 17, 3)
    """
    num_frames = len(frame_poses_list)
    num_joints = 17
    tracks = []
    num_tracks = 0

    for f_idx, poses in enumerate(frame_poses_list):
        if len(poses) == 0:
            continue

        active_proposals = [t for t in tracks if t['last_frame'] >= f_idx - max_frame_gap]
        n, m = len(active_proposals), len(poses)

        if n > 0 and m > 0:
            scores = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    scores[i, j] = dist_ske(active_proposals[i]['last_pose'], poses[j])

            row_ind, col_ind = linear_sum_assignment(scores)
            matched_cols = set()
            for r, c in zip(row_ind, col_ind):
                # Only accept match if skeleton distance is reasonable (< 1500 px)
                if scores[r, c] < 1500:
                    active_proposals[r]['frames'][f_idx] = poses[c]
                    active_proposals[r]['last_frame'] = f_idx
                    active_proposals[r]['last_pose'] = poses[c]
                    matched_cols.add(c)

            for j in range(m):
                if j not in matched_cols:
                    num_tracks += 1
                    new_track = {
                        'id': num_tracks,
                        'frames': {f_idx: poses[j]},
                        'last_frame': f_idx,
                        'last_pose': poses[j]
                    }
                    tracks.append(new_track)
        else:
            for pose in poses:
                num_tracks += 1
                new_track = {
                    'id': num_tracks,
                    'frames': {f_idx: pose},
                    'last_frame': f_idx,
                    'last_pose': pose
                }
                tracks.append(new_track)

    if not tracks:
        return np.zeros((max_tracks, num_frames, num_joints, 3), dtype=np.float32)

    # Sort tracks by number of observed frames (longest tracks first)
    tracks.sort(key=lambda t: -len(t['frames']))
    result = np.zeros((max_tracks, num_frames, num_joints, 3), dtype=np.float32)

    for i, track in enumerate(tracks[:max_tracks]):
        for f_idx, pose in track['frames'].items():
            result[i, f_idx] = pose

    return result


class PoseExtractor:
    """Extracts COCO 17-keypoint skeletons from video files using YOLOv8-Pose."""

    def __init__(self, model_name='yolov8n-pose.pt', device='cpu'):
        self.device = device
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)

    def extract_from_video(self, video_path, max_frames=300, target_short_side=480, progress_callback=None):
        """
        Extract video frames and track 2D body skeletons.
        Returns:
            dict containing:
                - frames: list of np.ndarray (RGB)
                - skeleton_sequence: np.ndarray of shape (num_person, T, 17, 3) (x, y, conf)
                - fps: float
                - duration: float
                - width: int
                - height: int
                - total_frames: int
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")

        self._load_model()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Compute resize dimensions maintaining aspect ratio
        if orig_w < orig_h:
            new_w = target_short_side
            new_h = int(orig_h * (target_short_side / orig_w))
        else:
            new_h = target_short_side
            new_w = int(orig_w * (target_short_side / orig_h))

        # Ensure dimensions are divisible by 2
        new_w = (new_w // 2) * 2
        new_h = (new_h // 2) * 2

        frames = []
        frame_poses = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_resized = cv2.resize(frame, (new_w, new_h))
            frames.append(frame_resized)
            frame_count += 1

            if max_frames and frame_count >= max_frames:
                break

        cap.release()

        if len(frames) == 0:
            raise ValueError("No frames could be extracted from the video.")

        # Batch / frame-wise YOLOv8-Pose inference
        if progress_callback:
            progress_callback(0.2, f"Detecting human poses across {len(frames)} frames...")

        for idx, frame in enumerate(frames):
            # Ultralytics inference
            results = self.model(frame, verbose=False, device=self.device)
            poses_in_frame = []

            for r in results:
                if r.keypoints is not None and len(r.keypoints.data) > 0:
                    kpts_data = r.keypoints.data.cpu().numpy()  # shape (num_persons, 17, 3)
                    for person_kpts in kpts_data:
                        # (17, 3) -> (x, y, conf)
                        poses_in_frame.append(person_kpts)

            frame_poses.append(poses_in_frame)

            if progress_callback and (idx % 10 == 0 or idx == len(frames) - 1):
                pct = 0.2 + 0.6 * (idx / len(frames))
                progress_callback(pct, f"Extracting skeletons: Frame {idx + 1}/{len(frames)}")

        # Perform tracking to construct (2, T, 17, 3) sequence
        skeleton_sequence = track_poses(frame_poses, max_tracks=2)
        duration = len(frames) / fps

        if progress_callback:
            progress_callback(0.85, "Pose extraction & tracking completed.")

        return {
            'frames': frames,
            'skeleton_sequence': skeleton_sequence,
            'fps': fps,
            'duration': duration,
            'width': new_w,
            'height': new_h,
            'total_frames': len(frames),
            'raw_poses': frame_poses
        }


class HandGestureExtractor:
    """
    Extracts 21-keypoint Hand landmarks from video for HaGRID gesture recognition.
    """

    def __init__(self):
        self.mp_hands = None
        self.hands_detector = None

    def _init_detector(self):
        if self.hands_detector is None:
            try:
                import mediapipe as mp
                self.mp_hands = mp.solutions.hands
                self.hands_detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception:
                self.hands_detector = None

    def extract_from_video(self, video_path, max_frames=150, progress_callback=None):
        """
        Extract hand keypoint sequence (1, T, 21, 2) from video.
        """
        self._init_detector()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        hand_kpts_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break

        cap.release()
        num_f = len(frames)

        if num_f == 0:
            raise ValueError("No frames extracted from gesture video.")

        if progress_callback:
            progress_callback(0.2, f"Extracting hand landmarks across {num_f} frames...")

        for idx, frame in enumerate(frames):
            kpt = np.zeros((21, 2), dtype=np.float32)
            if self.hands_detector is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.hands_detector.process(rgb)
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0]
                    kpt = np.array([[pt.x, pt.y] for pt in lm.landmark], dtype=np.float32)

            hand_kpts_list.append(kpt)
            if progress_callback and (idx % 10 == 0 or idx == num_f - 1):
                pct = 0.2 + 0.6 * (idx / num_f)
                progress_callback(pct, f"Hand tracking: Frame {idx + 1}/{num_f}")

        skeleton_sequence = np.array(hand_kpts_list)[None, ...]  # (1, T, 21, 2)

        return {
            'frames': frames,
            'skeleton_sequence': skeleton_sequence,
            'fps': fps,
            'duration': num_f / fps,
            'width': frames[0].shape[1],
            'height': frames[0].shape[0],
            'total_frames': num_f
        }
