"""Candidate tree -> OpenROAD route guides (gcell strips on the lowest H/V routing layers, pin access copied from
OpenROAD's own global-route guide), plus small geometry helpers. Same logic as the research code."""
import re
from collections import defaultdict
from pathlib import Path


def corridor_layers(geo, layer_range):
    lo, hi = layer_range.split('-'); L0, L1 = geo['layers'][lo]['level'], geo['layers'][hi]['level']
    inr = sorted((v['level'], k, v['direction']) for k, v in geo['layers'].items() if L0 <= v['level'] <= L1)
    h = next(k for _, k, d in inr if 'HOR' in d.upper()); v = next(k for _, k, d in inr if 'VER' in d.upper())
    return h, v, L0


def grt_pin_boxes(guide_file, net, geo, min_level):
    """Pin-access boxes GRT emitted for `net` on layers below the min routing layer (GRT's own pin-access gcells, so DRT accepts the guide)."""
    lines = Path(guide_file).read_text().split('\n'); out = []; i = 0
    while i < len(lines):
        if lines[i].strip() == net and i + 1 < len(lines) and lines[i + 1].strip() == '(':
            i += 2
            while lines[i].strip() != ')':
                x1, y1, x2, y2, layer = lines[i].split()
                if geo['layers'].get(layer, {}).get('level', 99) < min_level: out.append((int(x1), int(y1), int(x2), int(y2), layer))
                i += 1
            break
        i += 1
    return out


def guides_for(geom, pins, pin_boxes, grid, hl, vl):
    def gx(v): return max(0, min(grid['nx'] - 1, (int(v) - grid['x0']) // grid['sx']))
    def gy(v): return max(0, min(grid['ny'] - 1, (int(v) - grid['y0']) // grid['sy']))
    def box(cx, cy, l): return (grid['x0'] + cx * grid['sx'], grid['y0'] + cy * grid['sy'], grid['x0'] + (cx + 1) * grid['sx'], grid['y0'] + (cy + 1) * grid['sy'], l)
    cells = defaultdict(set); nodes = set()
    for x1, y1, x2, y2 in geom:
        nodes |= {(x1, y1), (x2, y2)}
        if y1 == y2:
            for c in range(gx(min(x1, x2)), gx(max(x1, x2)) + 1): cells[hl].add((c, gy(y1)))
        else:
            for c in range(gy(min(y1, y2)), gy(max(y1, y2)) + 1): cells[vl].add((gx(x1), c))
    node_cells = {(gx(x), gy(y)) for x, y in nodes | set(map(tuple, pins))}
    for c in node_cells: cells[hl].add(c); cells[vl].add(c)
    rects = []
    for x1, y1, x2, y2, l in pin_boxes:
        rects.append((x1, y1, x2, y2, l))
        for cx in range(gx(x1), gx(x2 - 1) + 1):
            for cy in range(gy(y1), gy(y2 - 1) + 1):
                nc = min(node_cells, key=lambda c: (abs(c[0] - cx) + abs(c[1] - cy), c))
                for ax in range(min(cx, nc[0]), max(cx, nc[0]) + 1):
                    for ay in range(min(cy, nc[1]), max(cy, nc[1]) + 1): cells[hl].add((ax, ay)); cells[vl].add((ax, ay))
    for layer, cs in cells.items():
        horizontal = layer == hl; lines = defaultdict(list)
        for cx, cy in cs: (lines[cy] if horizontal else lines[cx]).append(cx if horizontal else cy)
        for fixed, vals in sorted(lines.items()):
            vals = sorted(vals); start = prev = vals[0]
            for v in vals[1:] + [None]:
                if v is not None and v == prev + 1: prev = v; continue
                a = box(start, fixed, layer) if horizontal else box(fixed, start, layer)
                b = box(prev, fixed, layer) if horizontal else box(fixed, prev, layer)
                rects.append((a[0], a[1], b[2], b[3], layer))
                if v is not None: start = prev = v
    return rects


def window_nets(nets, rects, margin):
    x0 = min(r[0] for r in rects) - margin; y0 = min(r[1] for r in rects) - margin
    x1 = max(r[2] for r in rects) + margin; y1 = max(r[3] for r in rects) + margin
    return {n: s for n, s in nets.items() if any(r[0] <= x1 and r[2] >= x0 and r[1] <= y1 and r[3] >= y0 for _, _, r in s)}


def violations(rpt):
    """[(type, srcs, bbox-line)] from a DRT -output_drc report."""
    if not Path(rpt).exists(): return []
    out = []; cur = {}
    for ln in Path(rpt).read_text(errors='replace').splitlines():
        s = ln.strip()
        if s.startswith('violation type:'): cur = dict(t=s.split(':', 1)[1].strip())
        elif s.startswith('srcs:'): cur['s'] = s.split(':', 1)[1].strip()
        elif s.startswith('bbox ='):
            out.append((cur.get('t', ''), cur.get('s', ''), s.split('=', 1)[1].strip())); cur = {}
    return out


def gcell_grid(log):
    t = Path(log).read_text(errors='replace')
    gx = re.search(r'GCELLGRID X (-?\d+) DO (\d+) STEP (\d+)', t); gy = re.search(r'GCELLGRID Y (-?\d+) DO (\d+) STEP (\d+)', t)
    if not (gx and gy): return None
    return dict(x0=int(gx[1]), nx=int(gx[2]), sx=int(gx[3]), y0=int(gy[1]), ny=int(gy[2]), sy=int(gy[3]))
