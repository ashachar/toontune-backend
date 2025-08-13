#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centerline for COLOR cartoons with ridge filters + Hessian-directed NMS.

מה עושה:
1) בניית ערוץ בהירות יציב לצבע (L* או minRGB).
2) Frangi (או Sato) במולטי-סקייל עם black_ridges=True.
3) בחירת תגובת-מקסימום לכל פיקסל + מפה של sigma-best.
4) חישוב Hessian לכל sigma ובחירת רכיבי ההסיאן לפי ה-sigma-best לכל פיקסל.
5) NMS לאורך הנורמל לרכס (כמו ב-Canny, אבל בכיוון שמתקבל מה-Hessian).
6) סף, דילול (thinning) קצר ו-Pruning לענפים, פלט PNG שקוף; אופציונלי SVG (potrace).

תלויות: numpy, opencv-python, scikit-image
pip install numpy opencv-python scikit-image
"""
import os
import sys
import argparse
import numpy as np
import cv2

from skimage import exposure
from skimage.filters import frangi, sato, threshold_otsu
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.morphology import skeletonize

# ---------- עזרים ----------

def read_bgra(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img  # BGRA

def to_work_gray(bgra: np.ndarray, mode: str = "lab") -> np.ndarray:
    """יוצר ערוץ בהירות שבו דיו כהה נשאר כהה גם כשמסביב צבעוני."""
    b, g, r, a = cv2.split(bgra)
    bgr = cv2.merge([b, g, r])
    if mode == "minrgb":
        work = np.minimum(np.minimum(r, g), b).astype(np.float32) / 255.0
    else:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32) / 255.0
        work = L
    # רקע שקוף -> בהיר
    work = np.where(a > 0, work, 1.0).astype(np.float32)
    work = exposure.rescale_intensity(work, in_range="image", out_range=(0.0, 1.0)).astype(np.float32)
    return work

def ridge_multi(gray01: np.ndarray, sigmas, which: str = "frangi") -> tuple[np.ndarray, np.ndarray]:
    """מריץ Frangi/Sato לכל sigma ומחזיר (resp_max, idx_of_best)."""
    stack = []
    for s in sigmas:
        if which == "sato":
            resp = sato(gray01, sigmas=[s], black_ridges=True)
        else:
            resp = frangi(gray01, sigmas=[s], black_ridges=True)
        stack.append(resp.astype(np.float32))
    R = np.stack(stack, axis=-1)                  # HxWxK
    R = exposure.rescale_intensity(R, in_range="image", out_range=(0.0, 1.0))
    idx = np.argmax(R, axis=-1)                   # HxW -> argmax sigma
    resp = R.max(axis=-1).astype(np.float32)      # HxW
    return resp, idx

def pick_hessian_per_pixel(gray01: np.ndarray, sigmas, idx_best: np.ndarray):
    """מחשב Hessian לכל σ, ובוחר לכל פיקסל את רכיבי ההסיאן לפי ה-σ המתאים."""
    Hxx_list, Hxy_list, Hyy_list = [], [], []
    for s in sigmas:
        Hxx, Hxy, Hyy = hessian_matrix(gray01, sigma=s, order='rc')
        Hxx_list.append(Hxx); Hxy_list.append(Hxy); Hyy_list.append(Hyy)
    Hxxs = np.stack(Hxx_list, axis=-1)
    Hxys = np.stack(Hxy_list, axis=-1)
    Hyys = np.stack(Hyy_list, axis=-1)
    # אינדוקס רכיבי ה-Hessian לפי ה-sigma-winner לכל פיקסל
    rows, cols = np.indices(idx_best.shape)
    Hxx = Hxxs[rows, cols, idx_best]
    Hxy = Hxys[rows, cols, idx_best]
    Hyy = Hyys[rows, cols, idx_best]
    return Hxx, Hxy, Hyy

def orientation_normal_from_hessian(Hxx, Hxy, Hyy):
    """
    מחשב את כיוון הציר הראשי של העקמומיות (הנורמל לרכס) מתוך Hessian.
    זווית בכיוון הנורמל (רדיאנים, בטווח [0, π)).
    """
    # זווית האיגנווקטור של הערך העצמי הגדול במוחלט (כיוון הנורמל)
    # tan(2θ) = 2Hxy / (Hxx - Hyy)
    theta = 0.5 * np.arctan2(2.0 * Hxy, (Hxx - Hyy) + 1e-12)
    theta = np.mod(theta, np.pi)
    return theta

def nms_along_normal(resp: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    NMS פשוט ע"י קוונטיזציה ל-4 כיוונים (0°,45°,90°,135°) – דומה ל-Canny.
    שומר רק פיקסלים שהם מקסימום ביחס לשכנים לאורך הנורמל.
    """
    H, W = resp.shape
    out = np.zeros_like(resp, dtype=np.float32)
    # כימות כיוון הנורמל ל-4 סקטורים
    angle = (theta * 180.0 / np.pi) % 180.0
    sector = np.zeros_like(resp, dtype=np.uint8)
    sector[(angle < 22.5) | (angle >= 157.5)] = 0     # ~0°
    sector[(angle >= 22.5) & (angle < 67.5)] = 1      # ~45°
    sector[(angle >= 67.5) & (angle < 112.5)] = 2     # ~90°
    sector[(angle >= 112.5) & (angle < 157.5)] = 3    # ~135°

    # שכנים לפי סקטור
    shifts = {
        0: ((0, -1), (0, 1)),
        1: ((-1, 1), (1, -1)),
        2: ((-1, 0), (1, 0)),
        3: ((-1, -1), (1, 1)),
    }

    # פדינג לקצוות
    pad = np.pad(resp, 1, mode='edge')
    secp = np.pad(sector, 1, mode='edge')

    for s, ((dy1, dx1), (dy2, dx2)) in shifts.items():
        mask = (secp[1:-1, 1:-1] == s)
        r0 = resp[mask]
        n1 = pad[1 + dy1:H + 1 + dy1, 1 + dx1:W + 1 + dx1][mask]
        n2 = pad[1 + dy2:H + 1 + dy2, 1 + dx2:W + 1 + dx2][mask]
        keep = (r0 >= n1) & (r0 >= n2)
        out[mask] = np.where(keep, r0, 0.0)

    return out

def prune_short_branches(bin01: np.ndarray, min_len: int = 8, max_iters: int = 50) -> np.ndarray:
    """Pruning לענפים קצרים בשלד בינארי."""
    sk = bin01.copy().astype(np.uint8)
    if min_len <= 1:
        return sk
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(max_iters):
        neighbors = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT) - sk
        endpoints = ((sk == 1) & (neighbors == 1)).astype(np.uint8)
        if endpoints.sum() == 0:
            break
        changed = False
        to_remove = np.zeros_like(sk)
        frontier = endpoints.copy()
        for _step in range(min_len):
            if frontier.sum() == 0:
                break
            to_remove |= frontier
            tmp = (sk - to_remove).clip(0, 1)
            neigh = cv2.filter2D(tmp, -1, kernel, borderType=cv2.BORDER_CONSTANT) - tmp
            frontier = ((tmp == 1) & (neigh == 1) & (~to_remove.astype(bool))).astype(np.uint8)
            changed = True
        if not changed:
            break
        sk = (sk - to_remove).clip(0, 1).astype(np.uint8)
    return sk

def render_rgba(mask01: np.ndarray, H: int, W: int, color=(0, 0, 0)) -> np.ndarray:
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[:, :, :3] = (color[2], color[1], color[0])  # BGR
    rgba[:, :, 3] = (mask01 * 255).astype(np.uint8)
    return rgba

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Centerline for color cartoons (ridge + Hessian-NMS)")
    ap.add_argument("input", nargs="?", default="cartoon-test/robot_color.png")
    ap.add_argument("output", nargs="?", default="utils/contour_extraction/outline.png")
    ap.add_argument("--intensity", choices=["lab", "minrgb"], default="lab",
                    help="ערוץ בהירות: L* (lab) או minRGB")
    ap.add_argument("--filter", choices=["frangi", "sato"], default="frangi",
                    help="מסנן ridge (Frangi/Sato)")
    ap.add_argument("--sigmas", type=str, default="1,2,3,4,5,6,7,8,9,10",
                    help="טווח σ בפיקסלים (חצי רוחב סטְרוֹק משוער)")
    ap.add_argument("--nms", action="store_true", help="הפעל NMS לאורך הנורמל (מומלץ!)")
    ap.add_argument("--binarize", choices=["otsu", "fixed"], default="otsu",
                    help="שיטת סף על מפה לאחר NMS")
    ap.add_argument("--fixed-thr", type=float, default=0.25, help="סף קבוע אם binarize=fixed")
    ap.add_argument("--min-branch", type=int, default=8, help="Pruning לענפים קצרים")
    ap.add_argument("--stroke", choices=["black", "white"], default="black", help="צבע הקו בפלט")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input not found: {args.input}")
        sys.exit(1)

    print(f"[1/7] Read: {args.input}")
    bgra = read_bgra(args.input)
    H, W = bgra.shape[:2]

    print(f"[2/7] Build intensity from color: {args.intensity}")
    g = to_work_gray(bgra, mode=args.intensity)

    sigmas = tuple(float(s.strip()) for s in args.sigmas.split(",") if s.strip())
    print(f"[3/7] Ridge filter ({args.filter}) with sigmas={sigmas} (black_ridges=True)")
    resp, idx_best = ridge_multi(g, sigmas, which=args.filter)

    if args.nms:
        print("[4/7] Hessian at best sigma + NMS along normal")
        Hxx, Hxy, Hyy = pick_hessian_per_pixel(g, sigmas, idx_best)
        theta = orientation_normal_from_hessian(Hxx, Hxy, Hyy)
        resp = nms_along_normal(resp, theta)
    else:
        print("[4/7] (NMS disabled)")

    print("[5/7] Thresholding ridge map")
    if args.binarize == "otsu":
        thr = float(threshold_otsu(resp))
    else:
        thr = float(args.fixed_thr)
    bin_mask = (resp >= thr).astype(np.uint8)
    print(f"      thr={thr:.4f}, sum={int(bin_mask.sum())} pixels")

    print("[6/7] Thin + prune")
    skel = skeletonize(bin_mask.astype(bool)).astype(np.uint8)
    skel = prune_short_branches(skel, min_len=args.min_branch)

    color = (0, 0, 0) if args.stroke == "black" else (255, 255, 255)
    out = render_rgba(skel, H, W, color=color)
    
    # Only create directory if there is one
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(args.output, out)
    print(f"[7/7] Saved: {args.output}")

if __name__ == "__main__":
    main()
