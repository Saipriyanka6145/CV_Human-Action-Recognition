"""
PYSKL Inference Engine
Standalone PyTorch Graph Convolutional Network (ST-GCN / ST-GCN++)
for Action and Gesture Recognition, fully compatible with PYSKL checkpoints.
"""

import copy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os.path as osp

APP_DIR = osp.abspath(osp.dirname(__file__))
ROOT_DIR = osp.abspath(osp.join(APP_DIR, '..'))
for p in [ROOT_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from app.utils import OFFICIAL_CHECKPOINTS, get_checkpoint_path, load_label_map, get_available_device
except (ImportError, ModuleNotFoundError):
    from utils import OFFICIAL_CHECKPOINTS, get_checkpoint_path, load_label_map, get_available_device


# ==============================================================================
# SKELETON GRAPH TOPOLOGY (from PYSKL Graph Engine)
# ==============================================================================

def normalize_digraph(A, dim=0):
    """Normalize adjacency matrix."""
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.eye(num_node)
    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class SkeletonGraph:
    """Graph builder for COCO 17-keypoints and MediaPipe 21-hand keypoints."""

    def __init__(self, layout='coco', mode='spatial', max_hop=1):
        self.layout = layout
        self.mode = mode
        self.max_hop = max_hop

        if layout == 'coco':
            self.num_node = 17
            self.inward = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                (1, 0), (3, 1), (2, 0), (4, 2)
            ]
            self.center = 0
        elif layout == 'handmp':
            self.num_node = 21
            self.inward = [
                (1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7),
                (9, 0), (10, 9), (11, 10), (12, 11), (13, 0), (14, 13),
                (15, 14), (16, 15), (17, 0), (18, 17), (19, 18), (20, 19)
            ]
            self.center = 0
        elif layout == 'nturgb+d':
            self.num_node = 25
            neighbor_base = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)
            ]
            self.inward = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.center = 20
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)
        self.A = self.spatial()

    def spatial(self):
        Iden = edge2mat(self.self_link, self.num_node)
        In = normalize_digraph(edge2mat(self.inward, self.num_node))
        Out = normalize_digraph(edge2mat(self.outward, self.num_node))
        return np.stack((Iden, In, Out))


# ==============================================================================
# PYTORCH GCN MODULES (ST-GCN / ST-GCN++)
# ==============================================================================

class UnitGCN(nn.Module):
    """Unit Graph Convolutional Block."""

    def __init__(self, in_channels, out_channels, A, adaptive='init', conv_pos='pre', with_res=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)
        self.adaptive = adaptive
        self.conv_pos = conv_pos
        self.with_res = with_res

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        if self.adaptive == 'init':
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)

        if self.conv_pos == 'pre':
            self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        elif self.conv_pos == 'post':
            self.conv = nn.Conv2d(A.size(0) * in_channels, out_channels, 1)

        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = nn.Identity()

    def forward(self, x, A=None):
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0
        cur_A = self.A

        if self.conv_pos == 'pre':
            x = self.conv(x)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum('nkctv,kvw->nctw', (x, cur_A)).contiguous()
        elif self.conv_pos == 'post':
            x = torch.einsum('nctv,kvw->nkctw', (x, cur_A)).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x)

        return self.act(self.bn(x) + res)


class UnitTCN(nn.Module):
    """Unit Temporal Convolutional Block."""

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm='BN', dropout=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels) if norm == 'BN' else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))


class MSTCN(nn.Module):
    """Multi-Scale Temporal Convolutional Block (PYSKL ST-GCN++)."""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'], stride=1):
        super().__init__()
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            if isinstance(cfg, tuple) and cfg[0] == 'max':
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_c, kernel_size=1),
                    nn.BatchNorm2d(branch_c),
                    self.act,
                    nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))
                ))
                continue
            if isinstance(cfg, tuple) and isinstance(cfg[0], int):
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_c, kernel_size=1),
                    nn.BatchNorm2d(branch_c),
                    self.act,
                    UnitTCN(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None)
                ))

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels
        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels),
            self.act,
            nn.Conv2d(tin_channels, out_channels, kernel_size=1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        branch_outs = [tempconv(x) for tempconv in self.branches]
        feat = torch.cat(branch_outs, dim=1)
        return self.transform(feat)

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)


class STGCNBlock(nn.Module):
    """Combined Spatio-Temporal GCN Block."""

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__()
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}

        tcn_type = tcn_kwargs.pop('type', 'mstcn')
        self.gcn = UnitGCN(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = UnitTCN(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = MSTCN(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


class STGCNBackbone(nn.Module):
    """ST-GCN / ST-GCN++ Backbone matching exact PYSKL architecture."""

    def __init__(self, graph_cfg, in_channels=2, base_channels=64, ch_ratio=2,
                 num_stages=10, inflate_stages=[5, 8], down_stages=[5, 8], **kwargs):
        super().__init__()
        self.graph = SkeletonGraph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        lw_kwargs = [cp.deepcopy(kwargs) for _ in range(num_stages)]
        if len(lw_kwargs) > 0:
            lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        cur_channels = base_channels
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_c = cur_channels
            if i in inflate_stages:
                inflate_times += 1
            out_c = int(self.base_channels * (self.ch_ratio ** inflate_times) + 1e-4)
            cur_channels = out_c
            modules.append(STGCNBlock(in_c, out_c, A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)

    def forward(self, x):
        # x is (N, M, T, V, C)
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (N, M, V, C, T)
        x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])
        return x


class GCNClassificationHead(nn.Module):
    """GCN Classification Head."""

    def __init__(self, num_classes, in_channels, dropout=0.5):
        super().__init__()
        self.in_c = in_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)
        x = self.pool(x)
        x = x.reshape(N, M, C)
        x = x.mean(dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc_cls(x)


class PYSKLRecognizer(nn.Module):
    """Complete End-to-End PYSKL GCN Action Recognizer."""

    def __init__(self, backbone, cls_head):
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head

    def forward(self, keypoint):
        feat = self.backbone(keypoint)
        cls_score = self.cls_head(feat)
        return cls_score


# ==============================================================================
# MODEL FACTORY & INFERENCE PIPELINE
# ==============================================================================

def build_model(model_key='stgcnpp_ntu120', device='cpu'):
    """Build ST-GCN++ PyTorch model for specified dataset."""
    if model_key == 'stgcnpp_hagrid':
        backbone = STGCNBackbone(
            graph_cfg=dict(layout='handmp', mode='spatial'),
            in_channels=2,
            base_channels=64,
            ch_ratio=2,
            num_stages=6,
            down_stages=[6],
            inflate_stages=[6],
            gcn_adaptive='init',
            gcn_with_res=True,
            tcn_type='mstcn'
        )
        cls_head = GCNClassificationHead(num_classes=40, in_channels=128, dropout=0.0)
    else:  # NTU-120 action recognition
        backbone = STGCNBackbone(
            graph_cfg=dict(layout='coco', mode='spatial'),
            in_channels=3,
            base_channels=64,
            ch_ratio=2,
            num_stages=10,
            down_stages=[5, 8],
            inflate_stages=[5, 8],
            gcn_adaptive='init',
            gcn_with_res=True,
            tcn_type='mstcn'
        )
        cls_head = GCNClassificationHead(num_classes=120, in_channels=256, dropout=0.5)

    model = PYSKLRecognizer(backbone, cls_head)
    model.to(device)
    return model


def load_pyskl_model(model_key='stgcnpp_ntu120', device='cpu', progress_callback=None):
    """
    Build model, download checkpoint if necessary, load weights, and set eval mode.
    """
    ckpt_path = get_checkpoint_path(model_key, progress_callback=progress_callback)
    model = build_model(model_key, device=device)

    # Load PyTorch state dict
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # Strip prefixes if present
    cleaned_dict = {}
    for k, v in state_dict.items():
        k_clean = k
        if k_clean.startswith('module.'):
            k_clean = k_clean[7:]
        cleaned_dict[k_clean] = v

    model.load_state_dict(cleaned_dict, strict=True)
    model.eval()
    return model


def preprocess_skeleton_sequence(keypoint, keypoint_score=None, clip_len=100, num_person=2, layout='coco', in_channels=3):
    """
    Preprocess raw keypoints into PYSKL model tensor (1, M, T, V, C).
    Args:
        keypoint: np.ndarray of shape (M_raw, T_raw, V, 2 or 3)
        keypoint_score: np.ndarray of shape (M_raw, T_raw, V) or None
        clip_len: target temporal sequence length
        num_person: maximum number of persons to retain
        layout: 'coco' or 'handmp'
        in_channels: 2 or 3
    Returns:
        torch.Tensor of shape (1, num_person, clip_len, V, in_channels)
    """
    M_raw, T_raw, V = keypoint.shape[:3]
    
    # Format (x, y, score)
    if keypoint.shape[-1] == 2:
        if keypoint_score is not None:
            kp_full = np.concatenate([keypoint.astype(np.float32), keypoint_score[..., None].astype(np.float32)], axis=-1)
        else:
            default_score = np.ones((M_raw, T_raw, V, 1), dtype=np.float32)
            kp_full = np.concatenate([keypoint.astype(np.float32), default_score], axis=-1)
    else:
        kp_full = keypoint.copy().astype(np.float32)

    kp = kp_full[..., :in_channels]

    # 1. PreNormalize2D (centered and scaled to [-1, 1])
    for m in range(M_raw):
        valid = (kp[m, :, :, :2] != 0).any(axis=-1)
        if valid.sum() > 0:
            x_vals = kp[m, :, :, 0][valid]
            y_vals = kp[m, :, :, 1][valid]
            if len(x_vals) > 0 and len(y_vals) > 0:
                x_min, x_max = x_vals.min(), x_vals.max()
                y_min, y_max = y_vals.min(), y_vals.max()
                if (x_max - x_min) > 10 and (y_max - y_min) > 10:
                    kp[m, :, :, 0] = (kp[m, :, :, 0] - (x_max + x_min) / 2) / (x_max - x_min) * 2
                    kp[m, :, :, 1] = (kp[m, :, :, 1] - (y_max + y_min) / 2) / (y_max - y_min) * 2

    # 2. Uniform temporal sampling to clip_len
    if T_raw < clip_len:
        pad_len = clip_len - T_raw
        pad = np.repeat(kp[:, -1:], pad_len, axis=1)
        kp_sampled = np.concatenate([kp, pad], axis=1)
    elif T_raw > clip_len:
        indices = np.linspace(0, T_raw - 1, clip_len).astype(int)
        kp_sampled = kp[:, indices]
    else:
        kp_sampled = kp

    # 3. Format joint dimension (ensure V matches graph layout: 17 for coco, 21 for handmp)
    target_V = 21 if layout == 'handmp' else 17
    cur_V = kp_sampled.shape[2]
    if cur_V < target_V:
        pad_v = np.zeros((kp_sampled.shape[0], clip_len, target_V - cur_V, in_channels), dtype=np.float32)
        kp_sampled = np.concatenate([kp_sampled, pad_v], axis=2)
    elif cur_V > target_V:
        kp_sampled = kp_sampled[:, :, :target_V, :]

    # 4. Format person dimension (pad or truncate to num_person)
    if M_raw < num_person:
        pad_p = np.zeros((num_person - M_raw, clip_len, target_V, in_channels), dtype=np.float32)
        kp_sampled = np.concatenate([kp_sampled, pad_p], axis=0)
    elif M_raw > num_person:
        kp_sampled = kp_sampled[:num_person]

    # Shape: (num_person, clip_len, target_V, in_channels) -> (1, num_person, clip_len, target_V, in_channels) = (N, M, T, V, C)
    tensor = torch.from_numpy(kp_sampled).unsqueeze(0).float()
    return tensor


class PYSKLInferencePipeline:
    """High-level inference manager with caching and top-k output."""

    def __init__(self, model_key='stgcnpp_ntu120', device=None, progress_callback=None):
        self.model_key = model_key
        if device is None:
            self.device, self.device_name = get_available_device()
        else:
            self.device = device
            self.device_name = 'Custom Device'

        self.labels = load_label_map(model_key)
        self.model = load_pyskl_model(model_key, device=self.device, progress_callback=progress_callback)
        self.clip_len = 10 if model_key == 'stgcnpp_hagrid' else 100
        self.num_person = 1 if model_key == 'stgcnpp_hagrid' else 2
        self.layout = 'handmp' if model_key == 'stgcnpp_hagrid' else 'coco'
        self.in_channels = 2 if model_key == 'stgcnpp_hagrid' else 3

    def predict(self, skeleton_sequence, top_k=5, keypoint_score=None):
        """
        Execute prediction on skeleton sequence (M, T, V, 2 or 3).
        Returns:
            dict with:
                - top_action: str
                - top_confidence: float (0-100%)
                - predictions: list of (action_name, confidence_float)
                - raw_scores: np.ndarray of all softmax probabilities
        """
        tensor = preprocess_skeleton_sequence(
            skeleton_sequence,
            keypoint_score=keypoint_score,
            clip_len=self.clip_len,
            num_person=self.num_person,
            layout=self.layout,
            in_channels=self.in_channels
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)[0]
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_predictions = []
        for idx in top_indices:
            name = self.labels[idx] if idx < len(self.labels) else f"Class {idx}"
            conf = float(probs[idx]) * 100.0
            top_predictions.append((name, conf))

        return {
            'top_action': top_predictions[0][0],
            'top_confidence': top_predictions[0][1],
            'predictions': top_predictions,
            'raw_scores': probs,
            'top_indices': top_indices
        }
