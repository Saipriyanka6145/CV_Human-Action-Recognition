"""
Utility functions for device management, checkpoint downloading,
label loading, and temporary file management.
"""

import os
import os.path as osp
import shutil
import tempfile
import urllib.request
import torch

# Base directory for the project
BASE_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
CHECKPOINT_DIR = osp.join(BASE_DIR, 'checkpoints')
TEMP_DIR = osp.join(BASE_DIR, 'tmp')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Official PYSKL Pretrained Checkpoints
OFFICIAL_CHECKPOINTS = {
    'stgcnpp_ntu120': {
        'name': 'ST-GCN++ (NTU-RGB+D 120)',
        'type': 'action',
        'layout': 'coco',
        'num_classes': 120,
        'url': 'https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth',
        'filename': 'stgcnpp_ntu120_hrnet_j.pth',
        'label_map': osp.join(BASE_DIR, 'tools', 'data', 'label_map', 'nturgbd_120.txt'),
        'description': 'Graph Convolutional Network trained on 120 everyday actions (17 COCO keypoints)'
    },
    'stgcnpp_hagrid': {
        'name': 'ST-GCN++ (HaGRID Gestures)',
        'type': 'gesture',
        'layout': 'handmp',
        'num_classes': 40,
        'local_path': osp.join(BASE_DIR, 'demo', 'hagrid.pth'),
        'filename': 'hagrid.pth',
        'description': 'Real-time gesture recognition model trained on HaGRID dataset (21 Hand keypoints)'
    }
}

HAGRID_GESTURE_CLASSES = [
    'Doing other things', 'Drumming Fingers', 'No gesture',
    'Pulling Hand In', 'Pulling Two Fingers In', 'Pushing Hand Away', 'Pushing Two Fingers Away',
    'Rolling Hand Backward', 'Rolling Hand Forward', 'Shaking Hand',
    'Sliding Two Fingers Down', 'Sliding Two Fingers Left',
    'Sliding Two Fingers Right', 'Sliding Two Fingers Up',
    'Stop Sign', 'Swiping Down', 'Swiping Left', 'Swiping Right', 'Swiping Up',
    'Dislike', 'Like', 'Turning Hand Clockwise', 'Turning Hand Counterclockwise',
    'Zooming In With Full Hand', 'Zooming In With Two Fingers',
    'Zooming Out With Full Hand', 'Zooming Out With Two Fingers',
    'Call', 'Fist', 'Four', 'Mute', 'OK', 'One', 'Palm',
    'Peace', 'Rock', 'Three-Middle', 'Three-Left', 'Two Up', 'No Gesture'
]


def get_available_device():
    """Detect and return torch device string ('cuda:0' or 'cpu')."""
    if torch.cuda.is_available():
        return 'cuda:0', torch.cuda.get_device_name(0)
    return 'cpu', 'CPU (Host Processor)'


def load_label_map(model_key='stgcnpp_ntu120'):
    """Load action or gesture label list for a given model key."""
    if model_key == 'stgcnpp_hagrid':
        return HAGRID_GESTURE_CLASSES
    
    info = OFFICIAL_CHECKPOINTS.get(model_key, OFFICIAL_CHECKPOINTS['stgcnpp_ntu120'])
    label_path = info.get('label_map')
    if label_path and osp.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return [f'Action {i}' for i in range(info.get('num_classes', 120))]


def get_checkpoint_path(model_key='stgcnpp_ntu120', progress_callback=None):
    """
    Ensure checkpoint exists locally; download if needed.
    Returns absolute local filepath.
    """
    if model_key not in OFFICIAL_CHECKPOINTS:
        model_key = 'stgcnpp_ntu120'
    
    info = OFFICIAL_CHECKPOINTS[model_key]
    
    # Check if local path exists (e.g. demo/hagrid.pth)
    if 'local_path' in info and osp.exists(info['local_path']):
        return info['local_path']
    
    # Check if saved in checkpoints/
    dest_path = osp.join(CHECKPOINT_DIR, info['filename'])
    if osp.exists(dest_path) and os.path.getsize(dest_path) > 100000:
        return dest_path
    
    # Download from URL
    url = info.get('url')
    if not url:
        raise FileNotFoundError(f"Checkpoint for {model_key} not found and no download URL available.")
    
    if progress_callback:
        progress_callback(0.1, f"Downloading {info['name']} checkpoint (~5.9 MB)...")
    
    tmp_dest = dest_path + '.tmp'
    try:
        def reporthook(block_num, block_size, total_size):
            if total_size > 0 and progress_callback:
                percent = min(0.9, (block_num * block_size) / total_size)
                progress_callback(percent, f"Downloading {info['name']}: {int(percent*100)}%")

        urllib.request.urlretrieve(url, tmp_dest, reporthook=reporthook)
        shutil.move(tmp_dest, dest_path)
        if progress_callback:
            progress_callback(1.0, f"Checkpoint loaded: {info['name']}")
        return dest_path
    except Exception as e:
        if osp.exists(tmp_dest):
            os.remove(tmp_dest)
        raise RuntimeError(f"Failed to download checkpoint from {url}: {e}")


def create_temp_dir(prefix='pyskl_app_'):
    """Create a temporary directory for processing sessions."""
    return tempfile.mkdtemp(prefix=prefix, dir=TEMP_DIR)


def cleanup_temp_dir(dir_path):
    """Safely delete temporary directory and files."""
    if dir_path and osp.exists(dir_path):
        try:
            shutil.rmtree(dir_path, ignore_errors=True)
        except Exception:
            pass
