# center_zone.py
import cv2
import numpy as np

CENTER_POLY_FRAC = [
    (0.2475, 0.32),
    (0.652, 0.32),
    (0.688, 0.70),
    (0.21, 0.70),
]

def poly_from_frac(w: int, h: int, frac_points):
    pts = []
    for fx, fy in frac_points:
        pts.append([int(fx * w), int(fy * h)])
    return np.array(pts, dtype=np.int32)

def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, pt, False) >= 0

def bbox_triggers_zone(xyxy, poly, mode="overlap"):
    x1, y1, x2, y2 = map(int, xyxy)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    if mode == "center":
        return point_in_poly((cx, cy), poly)

    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (cx, cy)]
    return any(point_in_poly(c, poly) for c in corners)

