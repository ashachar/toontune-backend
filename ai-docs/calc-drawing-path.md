# SVG-First Stroke Traversal for Human-Like Drawing

## Quick Answer: SVG or PNG?

**Prefer SVG first.**

Work with the vector stroke geometry whenever possible:

**Pros (SVG):** 
- No thresholding/skeletonization noise
- Exact geometry
- Scale-invariant
- Easier to keep long, smooth, human-like strokes
- You can split paths precisely at intersections and traverse the resulting stroke graph

**Cons (SVG):** 
- Some files aren't "true strokes" (the author expanded strokes to filled outlines or used transforms/groups)
- You may need to flatten transforms and, if strokes were expanded, you'll treat them like filled shapes

**PNG is the fallback** when the SVG isn't usable as true strokes. PNG requires robust black-line isolation + skeletonization, but works on anything.

> **Recommendation used below:** SVG-first pipeline with a PNG fallback, both producing the same "draw sequence" JSON.

---

## What We'll Build

### 1. Extract stroke centerlines → a graph

- **SVG:** sample stroked paths to polylines, union + split at intersections (vector skeleton), build a graph.
- **PNG:** threshold near-black, clean, skeletonize to 1-pixel centerline, compress degree-2 runs, build a graph.

### 2. Make each connected component nearly Eulerian
Run a Chinese Postman augmentation to minimize pen-lifts by pairing odd-degree nodes along shortest paths.

### 3. Traverse edges like a human would
Angle-aware, bridge-avoiding traversal:
- Prefer non-bridge edges if available (don't burn bridges prematurely)
- Among candidates, choose minimum turning angle; if tied, take the longer edge

### 4. Output
JSON draw sequence (pen: "down" polylines + pen: "up" jumps) and a debug PNG render.

---

## Environment Check

**Python 3.10+**

Install: `numpy`, `opencv-python`, `scikit-image`, `networkx`, `svgpathtools`, `shapely`, `Pillow`

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install numpy opencv-python scikit-image networkx svgpathtools shapely pillow
```

> If your SVG uses transforms/groups: flatten them (e.g., re-save from Illustrator/Inkscape with transforms applied). That keeps the guide coherent.

---

## Usage

```bash
python stroke_traversal_svg_first.py input.svg out/run_svg
# or
python stroke_traversal_svg_first.py input.png out/run_png
```

**Outputs:**
- `out/run_*_draw.json` – ordered segments with pen: "down"/"up" and points
- `out/run_*_debug.png` – a quick visualization of the chosen stroke order

---

## Full Code (single file, complete): `stroke_traversal_svg_first.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SVG-first stroke traversal that "draws" a doodle like a human:

1) SVG pipeline:
   - Parse stroked, near-black paths, sample to polylines.
   - Use shapely.unary_union to split at all intersections (vector skeleton).
   - Build a graph: nodes at segment endpoints, edges as the split polylines.

2) PNG fallback:
   - Isolate near-black pixels in Lab, clean, skeletonize to 1px centerline.
   - Compress degree-2 pixel runs into polylines, build the same graph.

3) Chinese Postman augmentation per connected component:
   - Pair odd-degree nodes via min-weight matching, add duplicate edges along
     shortest paths → near-Eulerian multigraph.

4) Angle-aware, bridge-avoiding traversal:
   - Prefer non-bridge edges; pick smallest turn; if tie, the longer edge.
   - Emit draw segments with pen-up between disconnected components.

Outputs:
  - <prefix>_draw.json  (ordered segments)
  - <prefix>_debug.png  (render for sanity check)
"""

import os, sys, math, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import networkx as nx

# Images / skeleton
import cv2
from skimage.morphology import skeletonize
from PIL import Image, ImageDraw

# SVG + geometry
from svgpathtools import svg2paths2
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, linemerge

Point = Tuple[float, float]


# =========================
# Geometry helpers
# =========================
def euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def path_length(poly: List[Point]) -> float:
    if len(poly) < 2: return 0.0
    return sum(euclid(poly[i], poly[i+1]) for i in range(len(poly)-1))

def angle_of_segment(p: Point, q: Point) -> float:
    return math.atan2(q[1]-p[1], q[0]-p[0])

def angle_diff(a: float, b: float) -> float:
    # Absolute minimal difference between two angles
    d = (a - b + math.pi) % (2*math.pi) - math.pi
    return abs(d)

def simplify_polyline(poly: List[Point], eps: float = 0.35) -> List[Point]:
    # Slight simplification to remove jitter without changing shape
    if len(poly) < 3:
        return poly[:]
    ls = LineString(poly)
    ls2 = ls.simplify(eps, preserve_topology=False)
    return [(float(x), float(y)) for x, y in ls2.coords]


# =========================
# SVG -> vector skeleton graph
# =========================
def _is_near_black(stroke: Optional[str], black_tol: int = 40) -> bool:
    if not stroke or stroke in ('none', 'transparent'):
        return False
    s = stroke.strip().lower()
    if s.startswith('#'):
        hexv = s[1:]
        if len(hexv) == 3:
            r = int(hexv[0]*2, 16); g = int(hexv[1]*2, 16); b = int(hexv[2]*2, 16)
        elif len(hexv) == 6:
            r = int(hexv[0:2], 16); g = int(hexv[2:4], 16); b = int(hexv[4:6], 16)
        else:
            return False
        return (r < black_tol and g < black_tol and b < black_tol)
    if s.startswith('rgb'):
        nums = s[s.find('(')+1:s.find(')')].split(',')
        rgb = [int(float(v)) for v in nums[:3]]
        return all(v < black_tol for v in rgb)
    return False

def _hash_point(p: Point, tol: float = 1e-6) -> Tuple[int, int]:
    # Spatial hash to merge nearly-identical endpoints
    s = 1.0 / tol
    return (int(round(p[0] * s)), int(round(p[1] * s)))

def svg_to_graph_vector_skeleton(path_svg: str,
                                 black_tol: int = 40,
                                 sample_step_px: float = 2.0,
                                 simplify_eps: float = 0.35):
    """
    Build a stroke graph by sampling stroked paths, splitting at intersections
    via unary_union, and turning each split segment into an edge.

    Returns:
      G: networkx.Graph with node positions at 'pos'
      id2pt: node_id -> (x,y)
      edges_dict: {(min(u,v), max(u,v)): polyline}
      size: (W, H) for debug rendering
    """
    paths, attributes, svg_attr = svg2paths2(path_svg)

    # 1) Sample stroked, near-black paths into polylines
    polylines: List[List[Point]] = []
    for p, attr in zip(paths, attributes):
        stroke = attr.get('stroke', None)
        if not _is_near_black(stroke, black_tol=black_tol):
            continue

        length = p.length()
        # # of samples based on path length and desired step (≈ sample_step_px)
        samples = max(16, int(length / max(0.5, sample_step_px)))
        if samples > 10000:
            samples = 10000  # safety
        poly = []
        for i in range(samples+1):
            t = i / samples
            z = p.point(t)
            poly.append((float(z.real), float(z.imag)))
        poly = simplify_polyline(poly, eps=simplify_eps)
        if len(poly) >= 2:
            polylines.append(poly)

    if not polylines:
        raise ValueError("No near-black stroked paths found in SVG (or transforms not flattened).")

    # 2) Build shapely lines and split all intersections (vector "skeleton")
    lines = [LineString(pl) for pl in polylines]
    # unary_union splits at intersections and merges connectivity
    merged = unary_union(lines)

    # Collect split line segments
    split_segments: List[List[Point]] = []
    def _collect(ls):
        if isinstance(ls, LineString):
            coords = list(ls.coords)
            if len(coords) >= 2:
                split_segments.append([(float(x), float(y)) for (x,y) in coords])
        elif isinstance(ls, MultiLineString):
            for sub in ls:
                _collect(sub)
        else:
            # GeometryCollection, etc.: try to linemerge and recurse
            try:
                merged2 = linemerge(ls)
                _collect(merged2)
            except Exception:
                pass

    _collect(merged)

    if not split_segments:
        raise ValueError("Failed to obtain split segments from SVG (check input).")

    # 3) Create nodes at endpoints, edges as segments
    id2pt: Dict[int, Point] = {}
    pt2id: Dict[Tuple[int,int], int] = {}
    edges_dict: Dict[Tuple[int,int], List[Point]] = {}

    def get_node_id(pt: Point) -> int:
        key = _hash_point(pt, tol=1e-6)
        if key in pt2id:
            return pt2id[key]
        nid = len(id2pt)
        id2pt[nid] = (float(pt[0]), float(pt[1]))
        pt2id[key] = nid
        return nid

    for seg in split_segments:
        p0, p1 = seg[0], seg[-1]
        u = get_node_id(p0)
        v = get_node_id(p1)
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        poly = simplify_polyline(seg, eps=simplify_eps)
        edges_dict[key] = poly

    # 4) Build graph
    G = nx.Graph()
    for nid, pos in id2pt.items():
        G.add_node(nid, pos=pos)
    for (u, v), poly in edges_dict.items():
        G.add_edge(u, v, poly=poly, length=path_length(poly))

    # 5) Estimate canvas size for debug render
    xs = [p[0] for p in id2pt.values()]
    ys = [p[1] for p in id2pt.values()]
    if xs and ys:
        W = int(max(xs) - min(xs) + 20)
        H = int(max(ys) - min(ys) + 20)
    else:
        W, H = (1024, 768)

    return G, id2pt, edges_dict, (W, H)


# =========================
# PNG -> skeleton graph
# =========================
def png_to_skeleton_graph(path_png: str,
                          black_l_thresh: int = 60,
                          ab_thresh: int = 22,
                          simplify_eps: float = 0.35):
    """
    1) Near-black mask in Lab
    2) Clean (median + open)
    3) Skeletonize to 1px centerline
    4) Degree-2 chain compression to polylines
    """
    img_bgr = cv2.imread(path_png, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(path_png)
    H, W = img_bgr.shape[:2]

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img_lab)

    mask_black = (L < black_l_thresh) & (np.abs(A.astype(np.int16)-128) < ab_thresh) & (np.abs(B.astype(np.int16)-128) < ab_thresh)
    mask_black = mask_black.astype(np.uint8) * 255

    mask_black = cv2.medianBlur(mask_black, 3)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    skel = skeletonize((mask_black > 0)).astype(np.uint8)

    nbrs = [(-1,-1),(-1,0),(-1,1),
            (0,-1),        (0,1),
            (1,-1),(1,0),(1,1)]

    coords = np.argwhere(skel > 0)
    pix = set((int(r), int(c)) for r, c in coords)

    deg: Dict[Tuple[int,int], int] = {}
    for rc in pix:
        r, c = rc
        d = 0
        for dr, dc in nbrs:
            if (r+dr, c+dc) in pix:
                d += 1
        deg[rc] = d

    node_pixels = {rc for rc in pix if deg[rc] == 1 or deg[rc] >= 3}

    id2pt: Dict[int, Point] = {}
    pix2id: Dict[Tuple[int,int], int] = {}
    def add_node(rc):
        if rc in pix2id:
            return pix2id[rc]
        nid = len(id2pt)
        id2pt[nid] = (float(rc[1]), float(rc[0]))  # (x,y) = (col,row)
        pix2id[rc] = nid
        return nid

    def neighbors(rc):
        r, c = rc
        for dr, dc in nbrs:
            nr, nc = r+dr, c+dc
            if (nr, nc) in pix:
                yield (nr, nc)

    visited_dir = set()
    edges_dict: Dict[Tuple[int,int], List[Point]] = {}

    for start in list(node_pixels):
        for nxt in list(neighbors(start)):
            if (start, nxt) in visited_dir:
                continue
            if start not in pix or nxt not in pix:
                continue

            poly: List[Point] = [(float(start[1]), float(start[0]))]
            prev = start
            cur = nxt
            visited_dir.add((start, nxt)); visited_dir.add((nxt, start))

            while True:
                poly.append((float(cur[1]), float(cur[0])))
                dcur = deg.get(cur, 0)
                is_node = (dcur == 1) or (dcur >= 3) or (cur in node_pixels)
                if is_node: break
                nbrs2 = [p for p in neighbors(cur) if p != prev]
                if len(nbrs2) == 1:
                    nxt2 = nbrs2[0]
                    visited_dir.add((cur, nxt2)); visited_dir.add((nxt2, cur))
                    prev, cur = cur, nxt2
                else:
                    break

            u = add_node(start if start in node_pixels else prev)
            v = add_node(cur)
            if u == v: continue
            key = (u, v) if u < v else (v, u)
            edges_dict[key] = simplify_polyline(poly, eps=simplify_eps)

    G = nx.Graph()
    for nid, pos in id2pt.items():
        G.add_node(nid, pos=pos)
    for (u, v), poly in edges_dict.items():
        G.add_edge(u, v, poly=poly, length=path_length(poly))

    return G, id2pt, edges_dict, (W, H)


# =========================
# Chinese Postman augmentation
# =========================
def chinese_postman_augment(G: nx.Graph) -> Dict[Tuple[int,int], int]:
    """
    For each connected component:
    - Take odd-degree nodes
    - Build complete graph with edge weights = shortest path distance in G
    - Min-weight matching pairs the odds
    - Along each matched pair's shortest path, add "extra multiplicity" to edges
    Returns dict: extra[(min(u,v),max(u,v))] = count
    """
    extra: Dict[Tuple[int,int], int] = {}

    for comp_nodes in nx.connected_components(G):
        C = G.subgraph(comp_nodes).copy()
        odd = [n for n in C.nodes() if C.degree(n) % 2 == 1]
        if not odd:
            continue

        # Dijkstra from each odd node (small sets in doodles)
        from networkx.algorithms import matching
        K = nx.Graph()
        K.add_nodes_from(odd)

        dist_map: Dict[int, Dict[int, float]] = {}
        path_map: Dict[Tuple[int,int], List[int]] = {}
        for s in odd:
            dist, paths = nx.single_source_dijkstra(C, source=s, weight='length')
            dist_map[s] = dist
            for t in odd:
                if t == s: continue
                K.add_edge(s, t, weight=dist.get(t, float('inf')))
                if t in paths: path_map[(s,t)] = paths[t]

        M = matching.min_weight_matching(K, maxcardinality=True, weight='weight')

        for a, b in M:
            if (a,b) in path_map:
                nodes_path = path_map[(a,b)]
            elif (b,a) in path_map:
                nodes_path = path_map[(b,a)]
            else:
                nodes_path = nx.shortest_path(C, source=a, target=b, weight='length')

            for i in range(len(nodes_path)-1):
                u, v = nodes_path[i], nodes_path[i+1]
                key = (u, v) if u < v else (v, u)
                extra[key] = extra.get(key, 0) + 1

    return extra


# =========================
# Traversal (angle-aware + bridge-avoiding)
# =========================
@dataclass
class DrawSegment:
    pen: str
    points: List[Point]

def _angle_at_start(poly: List[Point], reversed_: bool) -> float:
    if len(poly) < 2:
        return 0.0
    if reversed_:
        p, q = poly[-1], poly[-2]
    else:
        p, q = poly[0], poly[1]
    return angle_of_segment(p, q)

def build_multigraph_with_extras(G: nx.Graph,
                                 extra: Dict[Tuple[int,int], int]) -> nx.MultiGraph:
    M = nx.MultiGraph()
    for n, data in G.nodes(data=True):
        M.add_node(n, **data)
    for u, v, data in G.edges(data=True):
        M.add_edge(u, v, poly=data['poly'], length=data['length'])
    for (u, v), k in extra.items():
        if k <= 0: continue
        poly = G.edges[u, v]['poly']
        length = G.edges[u, v]['length']
        for _ in range(k):
            M.add_edge(u, v, poly=poly, length=length)
    return M

def _residual_simple(M: nx.MultiGraph, remaining: Dict[Tuple[int,int,int], int]) -> nx.Graph:
    Gs = nx.Graph()
    Gs.add_nodes_from(M.nodes())
    for u, v, key in M.edges(keys=True):
        if remaining.get((u, v, key), 0) > 0:
            if not Gs.has_edge(u, v):
                Gs.add_edge(u, v)
    return Gs

def _multi_between_remaining(M: nx.MultiGraph, remaining, a: int, b: int) -> int:
    total = 0
    if a in M and b in M[a]:
        for key in M[a][b]:
            total += remaining.get((a, b, key), 0)
    return total

def traverse_angle_bridge_aware(M: nx.MultiGraph, id2pt: Dict[int, Point]) -> List[DrawSegment]:
    remaining: Dict[Tuple[int,int,int], int] = {}
    for u, v, key in M.edges(keys=True):
        remaining[(u, v, key)] = 1

    def nodes_with_remaining():
        ns = set()
        for (u, v, key), cnt in remaining.items():
            if cnt > 0:
                ns.add(u); ns.add(v)
        return list(ns)

    def avail_from(node: int):
        out = []
        if node not in M:
            return out
        for nbr in M.neighbors(node):
            for key in M[node][nbr].keys():
                if remaining.get((node, nbr, key), 0) > 0:
                    out.append((node, nbr, key))
        return out

    def is_bridge(a: int, b: int, key: int) -> bool:
        # Multiple parallel edges remaining => not a bridge
        if _multi_between_remaining(M, remaining, a, b) > 1:
            return False
        Gs = _residual_simple(M, remaining)
        if not Gs.has_edge(a, b):
            return False
        Gs.remove_edge(a, b)
        return not nx.has_path(Gs, a, b)

    def start_angle_for(a: int, b: int, key: int):
        poly = M.edges[a, b, key]['poly']
        pnode = id2pt[a]
        d0 = euclid(poly[0], pnode)
        d1 = euclid(poly[-1], pnode)
        if d0 <= d1:
            ang = _angle_at_start(poly, reversed_=False)
            return ang, True, poly
        else:
            ang = _angle_at_start(poly, reversed_=True)
            return ang, False, list(reversed(poly))

    segments: List[DrawSegment] = []
    cur: Optional[int] = None
    incoming: Optional[float] = None

    while any(cnt > 0 for cnt in remaining.values()):
        if cur is None:
            # Insert pen-up if last was down
            if segments and segments[-1].pen == "down":
                segments.append(DrawSegment(pen="up", points=[]))
            ns = nodes_with_remaining()
            if not ns: break
            # Prefer endpoints (degree 1 in residual), else leftmost
            Gs = _residual_simple(M, remaining)
            endpoints = [n for n in ns if Gs.degree(n) == 1]
            cur = sorted(endpoints, key=lambda n: (id2pt[n][0], id2pt[n][1]))[0] if endpoints else \
                  sorted(ns, key=lambda n: (id2pt[n][0], id2pt[n][1]))[0]
            incoming = None

        cands = []
        for (a, b, key) in avail_from(cur):
            ang, forward, oriented = start_angle_for(a, b, key)
            turn = 0.0 if incoming is None else angle_diff(incoming, ang)
            length = M.edges[a, b, key]['length']
            bridge = is_bridge(a, b, key)
            cands.append((bridge, turn, -length, b, key, forward, oriented, ang))

        if not cands:
            cur = None
            continue

        non_bridge = [c for c in cands if not c[0]]
        use = non_bridge if non_bridge else cands
        use.sort(key=lambda x: (x[1], x[2]))  # min turn, then longer

        _, _, _, b, key, forward, oriented, ang = use[0]
        remaining[(cur, b, key)] = 0
        segments.append(DrawSegment(pen="down", points=oriented))
        incoming = ang
        cur = b

    if segments and segments[-1].pen == "up":
        segments.pop()

    return segments


# =========================
# Debug render
# =========================
def save_debug_render(segments: List[DrawSegment], out_path: str,
                      size: Tuple[int,int], bg="white") -> None:
    W, H = size
    im = Image.new("RGB", (W, H), color=bg)
    draw = ImageDraw.Draw(im)
    for seg in segments:
        if seg.pen == "down" and len(seg.points) >= 2:
            draw.line(seg.points, width=2, fill=(0,0,0))
    im.save(out_path)


# =========================
# CLI
# =========================
def main():
    if len(sys.argv) < 3:
        print("Usage:\n  python stroke_traversal_svg_first.py input.svg output_prefix\n  python stroke_traversal_svg_first.py input.png output_prefix")
        sys.exit(1)

    in_path = sys.argv[1]
    out_prefix = sys.argv[2]
    ext = os.path.splitext(in_path)[1].lower()

    if ext == ".svg":
        G, id2pt, edges, size = svg_to_graph_vector_skeleton(
            in_path,
            black_tol=40,          # increase if strokes are gray
            sample_step_px=2.0,    # smaller -> denser sampling of curves
            simplify_eps=0.35
        )
    elif ext == ".png":
        G, id2pt, edges, size = png_to_skeleton_graph(
            in_path,
            black_l_thresh=60,     # increase if black is faint/anti-aliased
            ab_thresh=22,          # increase to tolerate slightly colored black
            simplify_eps=0.35
        )
    else:
        raise ValueError("Unsupported input (use .svg or .png).")

    if G.number_of_edges() == 0:
        print("No strokes detected. Check thresholds or input.")
        sys.exit(2)

    # Chinese Postman: add extra multiplicity to make components Eulerian
    extra = chinese_postman_augment(G)

    # Build multigraph with duplicates and traverse
    M = build_multigraph_with_extras(G, extra)
    segments = traverse_angle_bridge_aware(M, id2pt)

    # Write JSON
    out_json = out_prefix + "_draw.json"
    payload = [{"pen": s.pen, "points": [[float(x), float(y)] for (x,y) in s.points]} for s in segments]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Debug render
    out_png = out_prefix + "_debug.png"
    save_debug_render(segments, out_png, size=size)

    # Simple stats
    pen_lifts = sum(1 for s in segments if s.pen == "up")
    seg_down = sum(1 for s in segments if s.pen == "down")
    total_len = sum(path_length(s.points) for s in segments if s.pen == "down")
    print(f"[OK] Wrote: {out_json}")
    print(f"[OK] Wrote: {out_png}")
    print(f"Stats: draw_segments={seg_down}, pen_lifts={pen_lifts}, total_len≈{total_len:.1f}px")

if __name__ == "__main__":
    main()
```

---

## Why This Approach Yields Human-Like Drawing

- **Coverage guarantee:** Eulerian coverage (via Chinese Postman) ensures every stroke is covered with minimal duplication.
- **Few pen-lifts:** Pairing odd nodes in each component slashes jumps.
- **Hand feel:** At every junction, the minimum-turn heuristic preserves smooth direction; ties broken by longer edges favor long, flowing lines.
- **Bridge awareness:** Don't "burn" a bridge early—avoid edges that would disconnect the residual graph unless necessary.

---

## Tuning (and what to change when)

- **SVG sampling density:** `sample_step_px` ↓ (e.g., 1.0) for highly curved strokes; ↑ for speed.
- **SVG color tolerance:** `black_tol` ↑ if your stroke color is dark gray rather than pure black.
- **PNG black isolation:** raise `black_l_thresh` if antialiased blacks are being missed; raise `ab_thresh` if blacks are slightly colored.
- **Polyline simplification:** `simplify_eps` ~ 0.3–0.7 px is safe; larger smooths more but can shorten corners.

---

## Validation: Do We Really "Draw" Like a Human?

### Quantitative checks
- Pen-lifts count (lower is better)
- Average turn angle between consecutive edges (lower feels smoother)
- Average continuous run length (higher feels more human)

### Visual check
Animate the JSON: progressively reveal each "pen: down" polyline (e.g., SVG stroke-dashoffset or Canvas line drawing). You should see smooth, long strokes with few jumps.

---

## Common Pitfalls (with fixes)

### SVG looks empty
Paths used fills instead of strokes, or transforms weren't applied.
**Fix:** flatten transforms; ensure black/near-black stroke exists; otherwise use the PNG fallback.

### Messy intersections in SVG
The union/split step (`unary_union`) handles most cases. If your file has self-overlaps and micro-segments, slightly increase `simplify_eps` or the sampling density.

### PNG skeleton is noisy or breaks lines
Apply a tiny erode before skeletonization or smooth the mask more aggressively.

### Traversal zig-zags at T-junctions
Lower the weight of length (it's already secondary) or raise the penalty for large turns.

---

## When Should I Choose PNG on Purpose?

- Your SVG was exported with expanded outlines (fills), not strokes, and you can't flatten/re-stroke it
- You're mixing raster textures with the doodle lines
- You want the exact look of the raster edges (including anti-aliasing gaps)

In those cases, the PNG flow above (near-black → skeleton → graph) is the right hammer.

---

## Next Steps (optional improvements)

- **Exact SVG junctions:** We already split at all intersections in vector space. You can keep the original DOM z-order as a tiebreaker (authoring order) for even more "human" vibe.
- **Weighted chooser:** use a combined score `α·turn + β·(−length) + γ·bridge_penalty` and tune α,β,γ.
- **Output animated SVG:** Convert segments into `<path>` elements with stroke-dasharray/offset timelines.
