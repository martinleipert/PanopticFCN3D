import numpy as np


def box_iou(target, pred):
    """Compute 3D IoU between two axis-aligned bounding boxes: (x1, y1, z1, x2, y2, z2)."""
    pred_left, pred_top, pred_front, pred_right, pred_bottom, pred_back = np.split(pred, 6, axis=1)
    target_left, target_top, target_front, target_right, target_bottom, target_back = np.split(target, 6, axis=1)

    pred_vol = (pred_right - pred_left) * (pred_bottom - pred_top) * (pred_back - pred_front)
    target_vol = (target_right - target_left) * (target_bottom - target_top) * (target_back - target_front)

    w = np.minimum(pred_right, target_right) - np.maximum(pred_left, target_left)
    h = np.minimum(pred_bottom, target_bottom) - np.maximum(pred_top, target_top)
    d = np.minimum(pred_back, target_back) - np.maximum(pred_front, target_front)

    w = np.clip(w, a_min=0, a_max=None)
    h = np.clip(h, a_min=0, a_max=None)
    d = np.clip(d, a_min=0, a_max=None)

    inter_vol = w * h * d
    union_vol = pred_vol + target_vol - inter_vol

    iou = np.where(union_vol > 0, inter_vol / union_vol, 0.0)
    return iou


def binary_iou(target, prediction):
    """Compute binary IoU."""
    intersection = np.logical_and(target, prediction).sum()
    union = np.logical_or(target, prediction).sum()
    return intersection / union if union > 0 else 0.0

def binary_f1(target, prediction, return_count=True):
    """Compute binary F1 score."""
    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, np.logical_not(target)).sum()
    fn = np.logical_and(np.logical_not(prediction), target).sum()
    tn = np.logical_and(np.logical_not(prediction), np.logical_not(target)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    if return_count:
        return f1, (tp, fp, fn, tn)
    else:
        return f1
