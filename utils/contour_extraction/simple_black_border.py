#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI תואם ל-centerline_swt*: אותם דגלים ונתיבים.
מסמן גבול "דיו (כמעט שחור) ↔ לא-דיו" בתמונת PNG (כולל אלפא).
- לבן (255) = פיקסל גבול
- שחור (0)  = לא גבול

עדכון חשוב: opaque = (alpha >= 64) כדי *לא* לסמן קצה שחור↔שקוף עם אנטי־אליאס.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Tuple
import numpy as np
from PIL import Image

# ===== קבועים פנימיים (אין דגלי CLI, שמירת תאימות) =====
ALPHA_MIN = 64        # אלפא מינימלי להיחשב "לא שקוף" (היה 1)
NEAR_BLACK_MAX = 32   # max(R,G,B)<=32 נחשב "כמעט שחור"
LUMA_MAX = 56.0       # לומה נמוכה
SAT_MAX = 0.28        # רוויה נמוכה

# ------------ עזרי תמונה / מסכות ------------
def load_rgba(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGBA")
    return np.array(im, dtype=np.uint8)  # H x W x 4

def rgb_to_luma_and_sat(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r_f = r.astype(np.float32); g_f = g.astype(np.float32); b_f = b.astype(np.float32)
    cmax = np.maximum(np.maximum(r_f, g_f), b_f)
    cmin = np.minimum(np.minimum(r_f, g_f), b_f)
    luma = 0.2126 * r_f + 0.7152 * g_f + 0.0722 * b_f
    sat = (cmax - cmin) / (cmax + 1e-6)
    return luma, sat

def build_ink_and_color_masks(rgba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    ink_mask: "דיו" — כמעט שחור/אפור כהה אטום.
    color_mask: כל מה שאטום מספיק ואינו דיו.
    שקופים (alpha < ALPHA_MIN) לא נספרים בשום מסכה.
    """
    r, g, b, a = [rgba[..., i] for i in range(4)]
    opaque = (a >= ALPHA_MIN)

    luma, sat = rgb_to_luma_and_sat(r, g, b)
    near_black = (np.maximum(np.maximum(r, g), b) <= NEAR_BLACK_MAX)
    dark_grey  = (luma <= LUMA_MAX) & (sat <= SAT_MAX)

    ink_mask = opaque & (near_black | dark_grey)
    color_mask = opaque & (~ink_mask)
    return ink_mask, color_mask

def neighbors8(mask: np.ndarray) -> np.ndarray:
    up    = np.roll(mask,  1, axis=0); up[0, :] = False
    down  = np.roll(mask, -1, axis=0); down[-1, :] = False
    left  = np.roll(mask,  1, axis=1); left[:, 0] = False
    right = np.roll(mask, -1, axis=1); right[:, -1] = False
    uleft  = np.roll(up,   1, axis=1);  uleft[:, 0] = False
    uright = np.roll(up,  -1, axis=1);  uright[:, -1] = False
    dleft  = np.roll(down, 1, axis=1);  dleft[:, 0] = False
    dright = np.roll(down,-1, axis=1);  dright[:, -1] = False
    return up | down | left | right | uleft | uright | dleft | dright

def neighbor_count8(mask: np.ndarray) -> np.ndarray:
    total = np.zeros_like(mask, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0: continue
            rolled = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            if dy == -1: rolled[-1, :] = False
            if dy ==  1: rolled[0,  :] = False
            if dx == -1: rolled[:, -1] = False
            if dx ==  1: rolled[:,  0] = False
            total += rolled.astype(np.uint8)
    return total

def compute_boundary(ink_mask: np.ndarray, color_mask: np.ndarray) -> np.ndarray:
    # גבול = דיו עם שכן צבע או צבע עם שכן דיו (8-קישוריות)
    color_neigh = neighbors8(color_mask)
    ink_neigh   = neighbors8(ink_mask)
    boundary = (ink_mask & color_neigh) | (color_mask & ink_neigh)
    # ניקוי: הסרת פיקסלי גבול בודדים
    deg = neighbor_count8(boundary)
    return boundary & (deg >= 1)

def save_binary_mask(mask: np.ndarray, out_path: str):
    img = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    img.save(out_path)

# ------------ CLI תאימות + Debug ------------
def parse_canny_pair(s: str):
    try:
        a, b = s.split(","); return int(a), int(b)
    except Exception:
        raise argparse.ArgumentTypeError("CANNY format must be 'low,high' (e.g., 40,120)")

def ensure_debug_dir(path: str):
    if path and len(path.strip()) > 0:
        os.makedirs(path, exist_ok=True)

def save_debug(rgba, ink_mask, color_mask, boundary, debug_dir, args_dict, cwd):
    if not debug_dir: return
    ensure_debug_dir(debug_dir)
    with open(os.path.join(debug_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "cwd": cwd,
            "args": args_dict,
            "notes": "Opaque threshold set to alpha>=64 to ignore black↔transparent edges."
        }, f, ensure_ascii=False, indent=2)
    Image.fromarray(rgba, mode="RGBA").save(os.path.join(debug_dir, "00_input_rgba.png"))
    Image.fromarray(np.where(ink_mask, 255, 0).astype(np.uint8), mode="L").save(os.path.join(debug_dir, "01_mask_ink.png"))
    Image.fromarray(np.where(color_mask, 255, 0).astype(np.uint8), mode="L").save(os.path.join(debug_dir, "02_mask_color.png"))
    Image.fromarray(np.where(boundary, 255, 0).astype(np.uint8), mode="L").save(os.path.join(debug_dir, "03_boundary.png"))

def main():
    parser = argparse.ArgumentParser(description="Binary outline: ink(near-black) ↔ non-ink, ignoring low-alpha fringe. CLI-compatible with SWT-style flags.")
    parser.add_argument("input_png", type=str)
    parser.add_argument("output_png", type=str)
    # דגלי תאימות (לא בשימוש אלגוריתמי)
    parser.add_argument("--wmin", type=float, default=3.0)
    parser.add_argument("--wmax", type=float, default=28.0)
    parser.add_argument("--angle-tol", type=float, default=25.0)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--Lmax", type=float, default=60.0)
    parser.add_argument("--Cmax", type=float, default=22.0)
    parser.add_argument("--canny", type=parse_canny_pair, default=(40, 120))
    parser.add_argument("--grad-min", type=float, default=0.40)
    parser.add_argument("--inkcov-min", type=float, default=0.55)
    parser.add_argument("--accum-sigma", type=float, default=0.8)
    parser.add_argument("--accum-thr", type=float, default=0.18)
    parser.add_argument("--min-branch", type=int, default=10)
    parser.add_argument("--ring-fallback", action="store_true")
    parser.add_argument("--debug-dir", type=str, default=None)

    args = parser.parse_args()
    args_dict = vars(args).copy()

    cwd = os.getcwd()
    print("[centerline_swt_v4] cwd:", cwd)
    print("[centerline_swt_v4] input_png:", args.input_png)
    print("[centerline_swt_v4] output_png:", args.output_png)
    if args.debug_dir:
        print("[centerline_swt_v4] debug_dir:", args.debug_dir)

    rgba = load_rgba(args.input_png)
    ink_mask, color_mask = build_ink_and_color_masks(rgba)
    boundary = compute_boundary(ink_mask, color_mask)
    save_binary_mask(boundary, args.output_png)
    save_debug(rgba, ink_mask, color_mask, boundary, args.debug_dir, args_dict, cwd)

if __name__ == "__main__":
    main()
