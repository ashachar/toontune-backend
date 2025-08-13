#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centerline extraction for COLOR cartoons — Final (SWT-style, mutual pairing, orthogonality)
Default output: WHITE stroke on transparent background.

Key features:
  • Pair opposite edges (SWT-like) with STRICT mutual validation.
  • Enforce near-orthogonality of the chord to both edge normals.
  • Color gradient (Di Zenzo) for robust direction/magnitude on RGB.
  • Lab ink mask (low L*, low C*ab) to isolate neutral dark ink.
  • Accumulate midpoints → NMS (on accumulation) → skeletonize (1px).
  • Optional fallback: morphological ring (dilate−erode) + skeleton.
Dependencies: numpy, opencv-python, scikit-image (optional for skeletonize)
Install: pip install numpy opencv-python scikit-image
"""
import os, sys, math, argparse
from typing import Tuple
import numpy as np
import cv2

# optional, nicer skeleton
try:
    from skimage.morphology import skeletonize as sk_skeletonize
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False


# --------------------------- IO & color spaces ---------------------------

def read_bgra(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img  # BGRA


def minrgb_and_lab(bgra: np.ndarray):
    """Return intensity I01 in [0,1] (minRGB) and Lab image (float32)."""
    b, g, r, a = cv2.split(bgra)
    bgr = cv2.merge([b, g, r]).astype(np.uint8)
    I01 = np.minimum(np.minimum(r, g), b).astype(np.float32) / 255.0
    I01 = np.where(a > 0, I01, 1.0).astype(np.float32)  # transparent -> bright
    mn, mx = float(I01.min()), float(I01.max())
    if mx > mn: I01 = (I01 - mn) / (mx - mn)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return I01, lab


def lab_ink_mask(lab: np.ndarray, L_max: float, C_max: float) -> np.ndarray:
    """Binary mask for 'neutral dark ink': L*<=L_max and C*ab<=C_max."""
    L = lab[:, :, 0] * (100.0 / 255.0)     # OpenCV L in [0,255] -> [0,100]
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    C = np.sqrt(a * a + b * b) * (100.0 / 128.0)  # approx [0,100]
    return ((L <= L_max) & (C <= C_max)).astype(np.uint8)


# --------------------------- Di Zenzo color gradient ---------------------

def dizenzo_gradient(bgr: np.ndarray):
    """
    Compute color gradient via Di Zenzo:
      Ix/Iy per channel -> structure tensor -> orientation & magnitude.
    Returns: (theta [rad], grad_unit_x, grad_unit_y, magnitude)
    """
    Ix = []; Iy = []
    for c in range(3):
        ch = bgr[:, :, c]
        ix = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        iy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        Ix.append(ix); Iy.append(iy)
    Ix = np.stack(Ix, axis=-1)
    Iy = np.stack(Iy, axis=-1)
    Gxx = np.sum(Ix * Ix, axis=-1)
    Gxy = np.sum(Ix * Iy, axis=-1)
    Gyy = np.sum(Iy * Iy, axis=-1)

    theta = 0.5 * np.arctan2(2.0 * Gxy, (Gxx - Gyy) + 1e-12)
    tmp = np.sqrt(((Gxx - Gyy) * 0.5) ** 2 + (Gxy ** 2))
    lam_max = (Gxx + Gyy) * 0.5 + tmp
    mag = np.sqrt(np.maximum(lam_max, 0.0)) + 1e-6

    gx = np.cos(theta); gy = np.sin(theta)         # unit direction
    return theta, gx, gy, mag


# --------------------------- Edges (Canny from intensity) ----------------

def auto_canny_thresholds(gray8: np.ndarray, sigma: float = 0.33):
    med = np.median(gray8)
    lo = int(max(0, (1.0 - sigma) * med))
    hi = int(min(255, (1.0 + sigma) * med))
    lo = max(0, min(lo, hi - 1)); hi = min(255, max(hi, lo + 1))
    return lo, hi

def edges_canny_from_intensity(I01: np.ndarray, blur: int, canny: str):
    g8 = (np.clip(1.0 - I01, 0, 1) * 255).astype(np.uint8)  # ink->bright
    if blur > 0:
        g8 = cv2.GaussianBlur(g8, (2 * blur + 1, 2 * blur + 1), 0)
    if canny == "auto":
        lo, hi = auto_canny_thresholds(g8, sigma=0.33)
    else:
        lo, hi = [int(v) for v in canny.split(",")]
    e = cv2.Canny(g8, lo, hi, L2gradient=True)
    return (e > 0).astype(np.uint8)


# --------------------------- Helpers: thinning, NMS ----------------------

def skeletonize01(bin01: np.ndarray) -> np.ndarray:
    if SKIMAGE_OK:
        return sk_skeletonize(bin01.astype(bool)).astype(np.uint8)
    # Zhang–Suen fallback (compact implementation)
    img = bin01.copy().astype(np.uint8)
    prev = np.zeros_like(img)
    while True:
        P2 = np.roll(img, -1, 0);  P3 = np.roll(np.roll(img, -1, 0), 1, 1)
        P4 = np.roll(img, 1, 1);   P5 = np.roll(np.roll(img, 1, 0), 1, 1)
        P6 = np.roll(img, 1, 0);   P7 = np.roll(np.roll(img, 1, 0), -1, 1)
        P8 = np.roll(img, -1, 1);  P9 = np.roll(np.roll(img, -1, 0), -1, 1)
        neighbors = P2+P3+P4+P5+P6+P7+P8+P9
        trans = ((P2==0)&(P3==1)).astype(np.uint8)+((P3==0)&(P4==1)).astype(np.uint8)+ \
                ((P4==0)&(P5==1)).astype(np.uint8)+((P5==0)&(P6==1)).astype(np.uint8)+ \
                ((P6==0)&(P7==1)).astype(np.uint8)+((P7==0)&(P8==1)).astype(np.uint8)+ \
                ((P8==0)&(P9==1)).astype(np.uint8)+((P9==0)&(P2==1)).astype(np.uint8)
        m1 = (img==1)&(neighbors>=2)&(neighbors<=6)&(trans==1)&(P2*P4*P6==0)&(P4*P6*P8==0)
        img = img & (~m1)
        P2 = np.roll(img, -1, 0);  P3 = np.roll(np.roll(img, -1, 0), 1, 1)
        P4 = np.roll(img, 1, 1);   P5 = np.roll(np.roll(img, 1, 0), 1, 1)
        P6 = np.roll(img, 1, 0);   P7 = np.roll(np.roll(img, 1, 0), -1, 1)
        P8 = np.roll(img, -1, 1);  P9 = np.roll(np.roll(img, -1, 0), -1, 1)
        neighbors = P2+P3+P4+P5+P6+P7+P8+P9
        trans = ((P2==0)&(P3==1)).astype(np.uint8)+((P3==0)&(P4==1)).astype(np.uint8)+ \
                ((P4==0)&(P5==1)).astype(np.uint8)+((P5==0)&(P6==1)).astype(np.uint8)+ \
                ((P6==0)&(P7==1)).astype(np.uint8)+((P7==0)&(P8==1)).astype(np.uint8)+ \
                ((P8==0)&(P9==1)).astype(np.uint8)+((P9==0)&(P2==1)).astype(np.uint8)
        m2 = (img==1)&(neighbors>=2)&(neighbors<=6)&(trans==1)&(P2*P4*P8==0)&(P2*P6*P8==0)
        img = img & (~m2)
        if np.array_equal(img, prev): break
        prev = img.copy()
    return img.astype(np.uint8)


def nms_on_accum(acc: np.ndarray) -> np.ndarray:
    """
    4-direction NMS on an accumulation/ridge map (like Canny NMS).
    Returns a float32 map after suppression.
    """
    acc_blur = cv2.GaussianBlur(acc, (5,5), 0)
    gx = cv2.Sobel(acc_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(acc_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy) + 1e-6
    angle = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0

    out = np.zeros_like(acc_blur, dtype=np.float32)
    H, W = acc.shape
    # sector quantization
    sector = np.zeros_like(acc_blur, dtype=np.uint8)
    sector[(angle < 22.5) | (angle >= 157.5)] = 0
    sector[(angle >= 22.5) & (angle < 67.5)] = 1
    sector[(angle >= 67.5) & (angle < 112.5)] = 2
    sector[(angle >= 112.5) & (angle < 157.5)] = 3

    shifts = {0: ((0,-1),(0,1)), 1: ((-1,1),(1,-1)), 2: ((-1,0),(1,0)), 3: ((-1,-1),(1,1))}
    pad = np.pad(acc_blur, 1, mode='edge')
    secp = np.pad(sector, 1, mode='edge')

    for s, ((dy1,dx1),(dy2,dx2)) in shifts.items():
        mask = (secp[1:-1,1:-1] == s)
        r0 = acc_blur[mask]
        n1 = pad[1+dy1:H+1+dy1, 1+dx1:W+1+dx1][mask]
        n2 = pad[1+dy2:H+1+dy2, 1+dx2:W+1+dx2][mask]
        keep = (r0 >= n1) & (r0 >= n2)
        out[mask] = np.where(keep, r0, 0.0)
    return out


# --------------------------- SWT-like pairing ----------------------------

def normalize(vx, vy, eps=1e-6):
    mag = np.sqrt(vx*vx + vy*vy) + eps
    return vx/mag, vy/mag, mag

def bilinear_mean(arr01: np.ndarray, p: Tuple[int,int], q: Tuple[int,int], samples: int | None = None) -> float:
    y0,x0 = p; y1,x1 = q
    dx, dy = x1-x0, y1-y0
    dist = math.hypot(dx,dy)
    if samples is None: samples = max(2, int(dist*2))
    ts = np.linspace(0.0, 1.0, num=samples, dtype=np.float32)
    xs = x0 + ts*dx; ys = y0 + ts*dy
    xs = np.clip(xs, 0, arr01.shape[1]-1); ys = np.clip(ys, 0, arr01.shape[0]-1)
    x0i = np.floor(xs).astype(np.int32); x1i = np.minimum(x0i+1, arr01.shape[1]-1)
    y0i = np.floor(ys).astype(np.int32); y1i = np.minimum(y0i+1, arr01.shape[0]-1)
    wx = xs-x0i; wy = ys-y0i
    v00 = arr01[y0i, x0i]; v01 = arr01[y0i, x1i]; v10 = arr01[y1i, x0i]; v11 = arr01[y1i, x1i]
    vals = (1-wx)*(1-wy)*v00 + wx*(1-wy)*v01 + (1-wx)*wy*v10 + wx*wy*v11
    return float(np.mean(vals))


def pair_midpoints_strict(I01, ink01, ux, uy, gmag_color, edges01,
                          wmin, wmax, angle_tol_deg, step,
                          grad_min_value, inkcov_min,
                          enforce_mutual=True, enforce_ortho=True):
    """
    Pair opposite edges along ±normal with subpixel stepping.
    Enforce mutual pairing and near-orthogonality when requested.
    Returns: (accum_map float HxW, widthmap float HxW)
    """
    H, W = I01.shape
    cos_tol = math.cos(math.radians(angle_tol_deg))

    center_accum = np.zeros((H, W), dtype=np.float32)
    widthmap     = np.zeros((H, W), dtype=np.float32)
    visited      = np.zeros((H, W), dtype=np.uint8)

    edge_idx = np.argwhere(edges01 > 0)

    for (y, x) in edge_idx:
        if visited[y, x] or gmag_color[y, x] < grad_min_value:
            continue

        for sign in (+1, -1):
            dx = sign * ux[y, x]; dy = sign * uy[y, x]
            t = 0.0
            yi, xi = float(y), float(x)
            while t <= wmax:
                t += step
                yi = y + dy * t
                xi = x + dx * t
                if yi < 0 or yi >= H-1 or xi < 0 or xi >= W-1:
                    break
                yy, xx = int(round(yi)), int(round(xi))
                if edges01[yy, xx] == 0:
                    continue
                if gmag_color[yy, xx] < grad_min_value:
                    break
                # opposing normals?
                dot = dx * ux[yy, xx] + dy * uy[yy, xx]
                if dot > -cos_tol:
                    continue

                # distance and width gating
                dist = math.hypot(xx - x, yy - y)
                if dist < wmin or dist > wmax:
                    break

                # optional orthogonality: v parallel to n_p and -n_q
                if enforce_ortho:
                    vn = np.array([xx - x, yy - y], dtype=np.float32)
                    vn /= (np.linalg.norm(vn) + 1e-6)
                    np_dot = abs(vn[0] * ux[y, x] + vn[1] * uy[y, x])
                    nq_dot = abs((-vn[0]) * ux[yy, xx] + (-vn[1]) * uy[yy, xx])
                    if (np_dot < math.cos(math.radians(12.0))) or (nq_dot < math.cos(math.radians(12.0))):
                        continue

                # ink coverage along the chord
                cov = bilinear_mean(ink01.astype(np.float32), (y, x), (yy, xx))
                if cov < inkcov_min:
                    continue

                # optional mutual check: from q back to p
                if enforce_mutual:
                    ok_back = False
                    t2 = 0.0
                    while t2 <= wmax:
                        t2 += step
                        yb = yy - uy[yy, xx] * t2
                        xb = xx - ux[yy, xx] * t2
                        if int(round(yb)) == y and int(round(xb)) == x:
                            ok_back = True
                            break
                    if not ok_back:
                        continue

                # accept midpoint
                mx = int(round((x + xx) * 0.5)); my = int(round((y + yy) * 0.5))
                if 0 <= mx < W and 0 <= my < H:
                    center_accum[my, mx] += 1.0
                    widthmap[my, mx] = dist
                    visited[y, x] = 1; visited[yy, xx] = 1
                break
    return center_accum, widthmap


# --------------------------- Post-processing & fallback ------------------

def densify_nms_thin(accum: np.ndarray, sigma: float, thr_rel: float, min_branch: int) -> np.ndarray:
    """Blur accumulation, NMS, threshold, thin to 1px, prune short spurs."""
    if sigma > 0:
        k = max(1, int(3 * sigma)); k = 2 * k + 1
        acc = cv2.GaussianBlur(accum, (k, k), sigmaX=sigma)
    else:
        acc = accum.copy()
    acc_nms = nms_on_accum(acc)
    m = acc_nms.max()
    if m <= 0:
        return np.zeros_like(acc_nms, dtype=np.uint8)
    bin01 = (acc_nms >= (thr_rel * m)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin01 = cv2.morphologyEx(bin01, cv2.MORPH_CLOSE, kernel, iterations=1)
    skel = skeletonize01(bin01)
    # prune short spurs
    skel = prune_short_spurs(skel, min_len=min_branch)
    return skel


def prune_short_spurs(skel01: np.ndarray, min_len: int = 8, max_iters: int = 50) -> np.ndarray:
    sk = skel01.copy().astype(np.uint8)
    if min_len <= 1: return sk
    kernel = np.ones((3,3),np.uint8)
    for _ in range(max_iters):
        neighbors = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT) - sk
        endpoints = ((sk==1) & (neighbors==1)).astype(np.uint8)
        if endpoints.sum() == 0: break
        changed = False
        to_remove = np.zeros_like(sk)
        frontier = endpoints.copy()
        for _step in range(min_len):
            if frontier.sum() == 0: break
            to_remove |= frontier
            tmp = (sk - to_remove).clip(0,1)
            neigh = cv2.filter2D(tmp, -1, kernel, borderType=cv2.BORDER_CONSTANT) - tmp
            frontier = ((tmp==1) & (neigh==1) & (~to_remove.astype(bool))).astype(np.uint8)
            changed = True
        if not changed: break
        sk = (sk - to_remove).clip(0,1).astype(np.uint8)
    return sk


def ring_mat_fallback(ink01: np.ndarray, ring_r: int, min_branch: int):
    """Morphological gradient band on ink mask + skeleton."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*ring_r+1, 2*ring_r+1))
    dil = cv2.dilate(ink01, kernel, iterations=1)
    ero = cv2.erode (ink01, kernel, iterations=1)
    ring = ((dil > 0) & (ero == 0)).astype(np.uint8)
    skel = skeletonize01(ring)
    skel = prune_short_spurs(skel, min_len=min_branch)
    return skel, ring


def render_rgba(mask01: np.ndarray, H: int, W: int, color=(255,255,255)) -> np.ndarray:
    rgba = np.zeros((H,W,4),dtype=np.uint8)
    rgba[:,:,:3] = (color[2], color[1], color[0])  # B,G,R
    rgba[:,:,3]  = (mask01*255).astype(np.uint8)
    return rgba


# --------------------------- CLI ----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Centerline for color cartoons — Final SWT-like (mutual+ortho)")
    ap.add_argument("input",  nargs="?", default="cartoon-test/robot_color.png")
    ap.add_argument("output", nargs="?", default="utils/contour_extraction/outline.png")

    # preprocessing / edges
    ap.add_argument("--blur", type=int, default=1, help="Gaussian blur radius for Canny (px)")
    ap.add_argument("--canny", type=str, default="40,120", help="Canny 'low,high' or 'auto'")

    # Lab ink mask parameters
    ap.add_argument("--Lmax", type=float, default=60.0, help="Lab L* max (0..100)")
    ap.add_argument("--Cmax", type=float, default=20.0, help="Lab C*ab max (0..100)")

    # SWT pairing params
    ap.add_argument("--wmin", type=float, default=3.0, help="Min stroke width (px)")
    ap.add_argument("--wmax", type=float, default=40.0, help="Max stroke width (px)")
    ap.add_argument("--angle-tol", type=float, default=12.0, help="Opposite gradient angle tolerance (deg)")
    ap.add_argument("--step", type=float, default=0.5, help="March step along normal (subpixel)")
    ap.add_argument("--grad-min", type=float, default=0.30,
                    help="If in [0,1], use as quantile over color-gradient on edges; if >1, absolute value.")
    ap.add_argument("--inkcov-min", type=float, default=0.60, help="Min mean ink-mask coverage along chord (0..1)")
    ap.add_argument("--strict", action="store_true",
                    help="Enforce mutual pairing and orthogonality checks (recommended)")

    # accumulation → thin
    ap.add_argument("--accum-sigma", type=float, default=1.0, help="Gaussian sigma for midpoint accumulation")
    ap.add_argument("--accum-thr", type=float, default=0.20, help="Relative threshold of accumulation (0..1)")
    ap.add_argument("--min-branch", type=int, default=10,  help="Prune spur length (px)")

    # fallback
    ap.add_argument("--ring-fallback", action="store_true", help="If sparse, use ring-MAT fallback")
    ap.add_argument("--ring-r", type=int, default=2, help="Ring radius for morphological gradient")

    # debug
    ap.add_argument("--debug-dir", type=str, default="", help="Directory to write intermediate images")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input not found: {args.input}"); sys.exit(1)

    # ---------- load & masks ----------
    bgra = read_bgra(args.input)
    H, W = bgra.shape[:2]
    bgr = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGRA2BGR)
    I01, lab = minrgb_and_lab(bgra)
    ink01 = lab_ink_mask(lab, L_max=args.Lmax, C_max=args.Cmax)

    # ---------- edges ----------
    edges01 = edges_canny_from_intensity(I01, blur=args.blur, canny=args.canny)

    # ---------- color gradient (Di Zenzo) ----------
    theta, gx_u, gy_u, gmag_color = dizenzo_gradient(bgr)
    ux, uy, _ = normalize(gx_u, gy_u)
    if 0.0 <= args.grad_min <= 1.0:
        mags_on_edges = gmag_color[edges01 > 0]
        grad_min_value = float(np.quantile(mags_on_edges, args.grad_min)) if mags_on_edges.size else 0.0
    else:
        grad_min_value = float(args.grad_min)

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(args.debug_dir, "ink_mask.png"), (ink01*255).astype(np.uint8))
        cv2.imwrite(os.path.join(args.debug_dir, "edges.png"), (edges01*255).astype(np.uint8))

    # ---------- pairing ----------
    accum, widthmap = pair_midpoints_strict(
        I01, ink01, ux, uy, gmag_color, edges01,
        wmin=int(round(args.wmin)), wmax=int(round(args.wmax)),
        angle_tol_deg=args.angle_tol, step=args.step,
        grad_min_value=grad_min_value, inkcov_min=args.inkcov_min,
        enforce_mutual=args.strict, enforce_ortho=args.strict
    )

    if args.debug_dir:
        acc_vis = (accum / (accum.max()+1e-6) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.debug_dir, "center_accum.png"), acc_vis)

    # ---------- densify → thin ----------
    skel01 = densify_nms_thin(accum, sigma=args.accum_sigma, thr_rel=args.accum_thr, min_branch=args.min_branch)

    # ---------- fallback ----------
    if args.ring_fallback and skel01.sum() < (H * W * 0.0005):
        print("[i] SWT centerline is sparse → using ring fallback.")
        skel01, ring = ring_mat_fallback(ink01, ring_r=args.ring_r, min_branch=args.min_branch)
        if args.debug_dir:
            cv2.imwrite(os.path.join(args.debug_dir, "ring.png"), (ring*255).astype(np.uint8))

    # ---------- render WHITE ----------
    rgba = render_rgba(skel01, H, W, color=(255,255,255))
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(args.output, rgba)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
