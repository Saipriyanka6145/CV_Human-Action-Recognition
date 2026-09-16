"""
Movement & Biomechanical Motion Analytics Engine
Computes joint displacement, instantaneous velocities, kinetic motion index,
body quadrant engagement, and temporal motion metrics.
"""

import numpy as np


JOINT_NAMES_COCO = [
    'Nose', 'L Eye', 'R Eye', 'L Ear', 'R Ear',
    'L Shoulder', 'R Shoulder', 'L Elbow', 'R Elbow',
    'L Wrist', 'R Wrist', 'L Hip', 'R Hip',
    'L Knee', 'R Knee', 'L Ankle', 'R Ankle'
]

BODY_PARTS = {
    'Head / Face': [0, 1, 2, 3, 4],
    'Upper Body & Arms': [5, 6, 7, 8, 9, 10],
    'Torso & Hips': [11, 12],
    'Lower Body & Legs': [13, 14, 15, 16]
}


class MovementAnalytics:
    """Computes derived kinematic and kinetic motion analytics from skeleton sequences."""

    def __init__(self, skeleton_sequence, fps=25.0, layout='coco'):
        """
        Args:
            skeleton_sequence: np.ndarray of shape (M, T, V, 2 or 3)
            fps: video frames per second
            layout: 'coco' or 'handmp'
        """
        self.skeleton = skeleton_sequence.copy()
        self.fps = max(fps, 1.0)
        self.dt = 1.0 / self.fps
        self.layout = layout

        self.num_persons = self.skeleton.shape[0]
        self.num_frames = self.skeleton.shape[1]
        self.num_joints = self.skeleton.shape[2]

        # Use primary person (person 0) for detailed single-subject kinematics
        self.primary_kpts = self.skeleton[0, :, :, :2]  # shape (T, V, 2)

    def compute_joint_displacements(self):
        """
        Compute frame-by-frame Euclidean displacement for each joint.
        Returns:
            np.ndarray of shape (T-1, V)
        """
        if self.num_frames <= 1:
            return np.zeros((1, self.num_joints), dtype=np.float32)

        # Delta coordinates between consecutive frames
        delta = np.diff(self.primary_kpts, axis=0)  # (T-1, V, 2)
        displacements = np.linalg.norm(delta, axis=-1)  # (T-1, V)
        return displacements

    def compute_joint_velocities(self):
        """
        Compute joint velocity in pixels/sec over time.
        Returns:
            np.ndarray of shape (T-1, V)
        """
        disp = self.compute_joint_displacements()
        return disp / self.dt

    def compute_motion_intensity(self):
        """
        Compute temporal aggregate motion intensity index across all joints (0-100 scale).
        Returns:
            np.ndarray of shape (T-1,)
        """
        disp = self.compute_joint_displacements()
        raw_intensity = np.mean(disp, axis=-1)  # average displacement across joints

        # Normalize to 0-100 scale
        max_val = np.max(raw_intensity) if len(raw_intensity) > 0 and np.max(raw_intensity) > 0 else 1.0
        normalized = (raw_intensity / max_val) * 100.0 if max_val > 0 else raw_intensity
        return normalized

    def compute_body_part_activity(self):
        """
        Compute relative motion contribution (% of total movement) across body quadrants.
        Returns:
            dict of {body_part_name: percentage_float}
        """
        if self.layout != 'coco':
            return {'Hand Joints': 100.0}

        disp = self.compute_joint_displacements()  # (T-1, 17)
        total_disp_per_joint = np.sum(disp, axis=0)  # (17,)
        total_disp_all = np.sum(total_disp_per_joint)

        if total_disp_all <= 1e-4:
            return {part: 25.0 for part in BODY_PARTS}

        activity_breakdown = {}
        for part_name, joint_indices in BODY_PARTS.items():
            valid_indices = [idx for idx in joint_indices if idx < self.num_joints]
            part_sum = np.sum(total_disp_per_joint[valid_indices])
            pct = float((part_sum / total_disp_all) * 100.0)
            activity_breakdown[part_name] = round(pct, 2)

        return activity_breakdown

    def compute_summary_metrics(self, processing_time_sec=0.0):
        """
        Generate full structured analytics summary dictionary.
        """
        disp = self.compute_joint_displacements()
        vel = self.compute_joint_velocities()
        intensity = self.compute_motion_intensity()

        avg_velocity = float(np.mean(vel)) if len(vel) > 0 else 0.0
        peak_velocity = float(np.max(vel)) if len(vel) > 0 else 0.0
        avg_intensity = float(np.mean(intensity)) if len(intensity) > 0 else 0.0

        # Most active joint
        joint_total_move = np.sum(disp, axis=0) if len(disp) > 0 else np.zeros(self.num_joints)
        most_active_idx = int(np.argmax(joint_total_move))
        if self.layout == 'coco' and most_active_idx < len(JOINT_NAMES_COCO):
            most_active_joint = JOINT_NAMES_COCO[most_active_idx]
        else:
            most_active_joint = f"Joint #{most_active_idx}"

        # Kinetic energy proxy (sum of squared velocities)
        kinetic_energy_index = float(np.mean(vel ** 2) / 1000.0) if len(vel) > 0 else 0.0

        return {
            'total_frames': self.num_frames,
            'duration_sec': round(self.num_frames / self.fps, 2),
            'fps': round(self.fps, 2),
            'processing_time_sec': round(processing_time_sec, 3),
            'processing_fps': round(self.num_frames / processing_time_sec, 1) if processing_time_sec > 0 else 0.0,
            'num_tracked_persons': self.num_persons,
            'avg_joint_velocity_px_s': round(avg_velocity, 2),
            'peak_joint_velocity_px_s': round(peak_velocity, 2),
            'avg_motion_intensity': round(avg_intensity, 2),
            'kinetic_energy_index': round(kinetic_energy_index, 2),
            'most_active_joint': most_active_joint,
            'body_part_activity': self.compute_body_part_activity(),
            'timeline_motion_intensity': intensity.tolist(),
            'timeline_displacements': disp
        }
