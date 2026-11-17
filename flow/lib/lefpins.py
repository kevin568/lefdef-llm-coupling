"""Pin access for cells whose signal pins have shapes only on device layers (poly / diffusion), as in analog
transistor LEFs: OpenROAD's global router stops on them (GRT-0029 "Pin I56/G does not have geometries below the max
routing layer"). A copy of such a LEF (out/.cache/) gets, for every signal pin (not USE POWER / GROUND) without a shape
on a routing layer, a rectangle on the lowest routing layer over each of its shapes (at least one routing pitch wide,
so a via can land there): where the contact to that pin sits. The original files are never changed."""
import hashlib
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHAPE = re.compile(r'^\s*(RECT|POLYGON)\b(?:\s+MASK\s+\d+)?((?:\s+-?[\d.]+)+)\s*;', re.M)


def _layer_size(lefs, name):
    """(width, pitch) of a LEF layer in microns (0 when not given)"""
    for l in lefs:
        t = Path(l).read_text(errors='replace')
        m = re.search(r'^[ \t]*LAYER[ \t]+' + re.escape(name) + r'[ \t]*\r?\n(.*?)^[ \t]*END[ \t]+' + re.escape(name) + r'\b',
                      t, re.M | re.S)
        if m:
            w = re.search(r'^\s*WIDTH\s+([\d.]+)', m.group(1), re.M)
            p = re.search(r'^\s*PITCH\s+([\d.]+)', m.group(1), re.M)
            return float(w.group(1)) if w else 0., float(p.group(1)) if p else 0.
    return 0., 0.


def _grid(lefs):
    """MANUFACTURINGGRID in microns (0.001 when not given)"""
    for l in lefs:
        m = re.search(r'^\s*MANUFACTURINGGRID\s+([\d.]+)', Path(l).read_text(errors='replace'), re.M)
        if m and float(m.group(1)) > 0:
            return float(m.group(1))
    return 0.001


def _pin_fix(body, routing, base, side, grid=0.001):
    """new PIN body with access rectangles on layer base, or None when the pin needs none"""
    if re.search(r'^\s*USE\s+(POWER|GROUND)\b', body, re.M):
        return None
    ports = list(re.finditer(r'^([ \t]*)PORT\b[^\n]*\n(.*?)^[ \t]*END\b[ \t]*\r?$', body, re.M | re.S))
    if not ports:
        return None
    rects, on_routing = [], False
    for p in ports:
        layer = None
        for st in re.finditer(r'^\s*LAYER\s+(\S+)[^;]*;|' + SHAPE.pattern[1:], p.group(2), re.M):
            if st.group(0).lstrip().startswith('LAYER'):
                layer = st.group(1); continue
            if layer in routing:
                on_routing = True
            else:
                v = [float(x) for x in st.group(3).split()]
                xs, ys = v[0::2], v[1::2]
                rects.append((min(xs), min(ys), max(xs), max(ys)))
    if on_routing or not rects:
        return None
    lines = []
    import math
    dn = lambda v: math.floor(v / grid + 1e-6) * grid   # outward onto the manufacturing grid (DRT-0320 otherwise)
    up = lambda v: math.ceil(v / grid - 1e-6) * grid
    dec = max(0, -int(math.floor(math.log10(grid)))) + 1
    for x1, y1, x2, y2 in rects:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw, hh = max((x2 - x1) / 2, side / 2), max((y2 - y1) / 2, side / 2)
        lines.append('        RECT ' + ' '.join('%.*f' % (dec, v) for v in (dn(cx - hw), dn(cy - hh), up(cx + hw), up(cy + hh))) + ' ;')
    p = ports[0]
    ind = p.group(1)
    add = f'{ind}  LAYER {base} ;\n' + '\n'.join(lines) + '\n'
    return body[:p.end(2)] + add + body[p.end(2):]


def add_pin_access(lefs, routing):
    """-> (LEF paths, notes). routing: routing layer names, lowest first."""
    if not routing:
        return list(lefs), []
    base = routing[0]
    w, pitch = _layer_size(lefs, base)
    side = max(2 * w, pitch)
    grid = _grid(lefs)
    out, fixed = [], []
    for l in lefs:
        t = Path(l).read_text(errors='replace')
        if not re.search(r'^\s*MACRO\s', t, re.M):
            out.append(l); continue
        changed = [False]

        def macro(mm):
            name = mm.group(1); mbody = mm.group(2)

            def pin(pm):
                new = _pin_fix(pm.group(2), set(routing), base, side, grid)
                if new is None:
                    return pm.group(0)
                changed[0] = True; fixed.append(f'{name}/{pm.group(1)}')
                return pm.group(0).replace(pm.group(2), new, 1)
            nb = re.sub(r'^[ \t]*PIN[ \t]+(\S+)[ \t]*\r?\n(.*?)^[ \t]*END[ \t]+\1\b', pin, mbody, flags=re.M | re.S)
            return mm.group(0).replace(mbody, nb, 1) if nb != mbody else mm.group(0)
        nt = re.sub(r'^[ \t]*MACRO[ \t]+(\S+)[ \t]*\r?\n(.*?)^[ \t]*END[ \t]+\1\b', macro, t, flags=re.M | re.S)
        if not changed[0]:
            out.append(l); continue
        cache = Path(os.environ.get('LEFDEF_CACHE') or REPO / 'out' / '.cache')
        dst = cache / hashlib.sha256(t.encode(errors='replace')).hexdigest()[:16] / (Path(l).stem + '.access.lef')
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(nt)
        out.append(str(dst))
    notes = []
    if fixed:
        notes.append(f'{len(fixed)} signal pin(s) have shapes only on device layers (e.g. {", ".join(fixed[:3])}): '
                     f'a {base} shape is added over each (copy of the cell LEF) so the router can reach them')
    return out, notes
