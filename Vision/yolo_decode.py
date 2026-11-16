"""
Vision/yolo_decode.py: YOLO pre- and post-processing shared by every backend.

Moved from bench/hailo_run.py (cv/hailo), where it was checked against
Ultralytics on 40 COCO images for YOLOv8n and YOLO26n: 94% of boxes matched,
mean IoU 0.97, confidence within 0.004.

Layouts (picked from the output shape):
  raw : (4+nc, N) or (N, 4+nc): cx, cy, w, h, then one score per class. No
        objectness (W26's bug was decoding this as YOLOv5). Needs NMS.
  e2e : (K, 6): x1, y1, x2, y2, conf, class (YOLO26 end-to-end head).
  nms : Hailo on-chip NMS, per class rows of ymin, xmin, ymax, xmax, score (0..1).
  split: one box output (H, W, 4) and one class output (H, W, nc) per stride, as in
        W26's turkey_640_v4.hef. Boxes are left/top/right/bottom distances from the
        cell center in cells (YOLO26, no DFL); class values are logits (sigmoid here).

Why split matters (found on the robot 11-13): W26's single-output HEFs pack boxes
(0 to ~770 px) and class scores (0 to 1) into ONE uint8 tensor with one scale, so
every class score quantizes to exactly 0 and nothing is ever detected. The split
HEF keeps a separate scale per output, so the scores survive.
"""

import cv2
import numpy as np


def letterbox(img, size, pad=114):
    """Resize keeping aspect ratio and pad, as Ultralytics does. Returns image, scale, (padx, pady)."""
    h, w = img.shape[:2]
    th, tw = size
    r = min(th / h, tw / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padx, pady = (tw - nw) / 2, (th - nh) / 2
    top, left = int(round(pady - 0.1)), int(round(padx - 0.1))
    out = cv2.copyMakeBorder(resized, top, th - nh - top, left, tw - nw - left,
                             cv2.BORDER_CONSTANT, value=(pad, pad, pad))
    return out, r, (left, top)


def unletterbox(boxes, r, pad, orig_wh):
    """boxes (k,4) x1y1x2y2 in network pixels -> original image pixels, clipped."""
    if len(boxes) == 0:
        return boxes
    b = boxes.copy().astype(np.float32)
    b[:, [0, 2]] = (b[:, [0, 2]] - pad[0]) / r
    b[:, [1, 3]] = (b[:, [1, 3]] - pad[1]) / r
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, orig_wh[0])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, orig_wh[1])
    return b


# ---------------------------------------------------------------- decoding

def nms(boxes, scores, classes, iou=0.45):
    """Class-aware NMS in numpy. boxes (k,4) x1y1x2y2."""
    keep = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        idx = idx[np.argsort(-scores[idx])]
        while len(idx):
            i = idx[0]
            keep.append(i)
            if len(idx) == 1:
                break
            rest = idx[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            idx = rest[inter / (a + b - inter + 1e-9) < iou]
    keep = np.array(sorted(keep, key=lambda k: -scores[k]), dtype=int)
    return keep


def guess_layout(arr):
    if isinstance(arr, (list, tuple)):
        return "nms"
    a = np.squeeze(np.asarray(arr))
    if a.ndim != 2:
        raise ValueError(f"can't decode output of shape {np.asarray(arr).shape}; run `info`")
    if 6 in a.shape and min(a.shape) == 6 and max(a.shape) <= 1000:
        return "e2e"
    return "raw"


def decode(out, layout, net_wh, conf):
    """Returns boxes (k,4) x1y1x2y2 in network pixels, scores (k,), classes (k,)."""
    nw, nh = net_wh
    if layout == "nms":
        # out: [batch][class] -> array (k,5) ymin,xmin,ymax,xmax,score normalized
        per_class = out[0] if len(out) and isinstance(out[0], (list, tuple)) else out
        B, S, C = [], [], []
        for c, rows in enumerate(per_class):
            rows = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
            rows = rows[rows[:, 4] >= conf]
            for y1, x1, y2, x2, s in rows:
                B.append([x1 * nw, y1 * nh, x2 * nw, y2 * nh]); S.append(s); C.append(c)
        return np.array(B, np.float32).reshape(-1, 4), np.array(S, np.float32), np.array(C, int)

    a = np.squeeze(np.asarray(out, dtype=np.float32))
    if layout == "e2e":
        if a.shape[0] == 6 and a.shape[1] != 6:
            a = a.T                                  # -> (K, 6)
        a = a[a[:, 4] >= conf]
        boxes = a[:, :4]
        if boxes.size and boxes.max() <= 1.5:        # normalized coordinates
            boxes = boxes * np.array([nw, nh, nw, nh], np.float32)
        return boxes, a[:, 4], a[:, 5].astype(int)

    # raw: (4+nc, N) -> (N, 4+nc); the anchor axis is the long one
    if a.shape[0] < a.shape[1]:
        a = a.T
    xywh, cls_scores = a[:, :4], a[:, 4:]
    classes = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(a)), classes]
    m = scores >= conf
    xywh, scores, classes = xywh[m], scores[m], classes[m]
    if xywh.size and xywh.max() <= 1.5:
        xywh = xywh * np.array([nw, nh, nw, nh], np.float32)
    boxes = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
    k = nms(boxes, scores, classes)
    return boxes[k], scores[k], classes[k]


def decode_split(outputs, net_wh, conf, iou=0.45):
    """outputs: iterable of arrays shaped (H, W, 4) for boxes and (H, W, nc) for class
    logits, one pair per stride. Returns boxes (k,4) x1y1x2y2 in network pixels,
    scores (k,), classes (k,)."""
    groups = {}
    for v in outputs:
        a = np.squeeze(np.asarray(v, dtype=np.float32))
        if a.ndim != 3:
            raise ValueError(f"split output must be (H, W, C), got {a.shape}")
        h, w, c = a.shape
        groups.setdefault((h, w), {})["box" if c == 4 else "cls"] = a
    B, S, C = [], [], []
    for (h, w), g in groups.items():
        if "box" not in g or "cls" not in g:
            raise ValueError(f"stride {h}x{w} is missing its box or class output")
        stride = net_wh[0] / w
        prob = 1.0 / (1.0 + np.exp(-g["cls"]))
        sc, cl = prob.max(-1), prob.argmax(-1)
        ys, xs = np.nonzero(sc >= conf)
        if len(ys) == 0:
            continue
        ltrb = g["box"][ys, xs]
        cx, cy = xs + 0.5, ys + 0.5
        boxes = np.stack([cx - ltrb[:, 0], cy - ltrb[:, 1], cx + ltrb[:, 2], cy + ltrb[:, 3]], 1) * stride
        B.append(boxes); S.append(sc[ys, xs]); C.append(cl[ys, xs])
    if not B:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    boxes, scores, classes = np.concatenate(B), np.concatenate(S), np.concatenate(C).astype(int)
    k = nms(boxes, scores, classes, iou)
    return boxes[k], scores[k], classes[k]
