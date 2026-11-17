"""Pin access for cells whose signal pins the router cannot reach: shapes only on device layers (poly / diffusion,
as in analog transistor LEFs) or only above the top signal routing layer (e.g. a MIM capacitor plate). OpenROAD's
global router stops on them (GRT-0029 "Pin I56/G does not have geometries below the max routing layer").

A copy of such a LEF (out/.cache/) gets, for every such signal pin (not USE POWER / GROUND), a rectangle over each of
its shapes: on the lowest routing layer under device-layer shapes, on the top signal layer under shapes above it
(where the contact / via to that pin sits). Each rectangle grows from the pin's own outline towards one routing pitch
(so a via can land on it) but stops one layer spacing short of every other pin's shapes on that layer in the same cell
(overlapping pads of different pins are shorts: DRT-0085 "valid access pattern combination not found"). Coordinates
are snapped to the manufacturing grid (DRT-0320 offgrid otherwise). The original files are never changed."""
import hashlib
import math
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHAPE_RE = r'\s*(RECT|POLYGON)\b(?:\s+MASK\s+\d+)?((?:\s+-?[\d.]+)+)\s*;'
STMT = re.compile(r'^\s*LAYER\s+(\S+)[^;]*;|^' + SHAPE_RE, re.M)


def _layer_size(lefs, name):
    """(width, pitch, spacing) of a LEF layer in microns (0 when not given)"""
    for l in lefs:
        t = Path(l).read_text(errors='replace')
        m = re.search(r'^[ \t]*LAYER[ \t]+' + re.escape(name) + r'[ \t]*\r?\n(.*?)^[ \t]*END[ \t]+' + re.escape(name) + r'\b',
                      t, re.M | re.S)
        if m:
            g = lambda k: re.search(r'^\s*' + k + r'\s+([\d.]+)', m.group(1), re.M)
            w, p, s = g('WIDTH'), g('PITCH'), g('SPACING')
            return tuple(float(x.group(1)) if x else 0. for x in (w, p, s))
    return 0., 0., 0.


def _grid(lefs):
    """MANUFACTURINGGRID in microns (0.001 when not given)"""
    for l in lefs:
        m = re.search(r'^\s*MANUFACTURINGGRID\s+([\d.]+)', Path(l).read_text(errors='replace'), re.M)
        if m and float(m.group(1)) > 0:
            return float(m.group(1))
    return 0.001


def _shapes(text):
    """[(layer, (x1, y1, x2, y2))] of the RECT / POLYGON statements in a PIN / OBS body"""
    out, layer = [], None
    for st in STMT.finditer(text):
        if st.group(1):
            layer = st.group(1); continue
        v = [float(x) for x in st.group(3).split()]
        if len(v) >= 4 and layer:
            out.append((layer, (min(v[0::2]), min(v[1::2]), max(v[0::2]), max(v[1::2]))))
    return out


def _grow(r, side, others, space):
    """r grown towards side x side (centred), but kept >= space away from every rectangle in others"""
    x1, y1, x2, y2 = r
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    tw, th = max(x2 - x1, side), max(y2 - y1, side)
    best = r
    for k in range(1, 11):   # 10 steps from the pin outline to the full target size: keep the largest that fits
        w = (x2 - x1) + (tw - (x2 - x1)) * k / 10
        h = (y2 - y1) + (th - (y2 - y1)) * k / 10
        c = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        if any(c[0] < o[2] + space and o[0] < c[2] + space and c[1] < o[3] + space and o[1] < c[3] + space
               for o in others):
            break
        best = c
    return best


def add_pin_access(lefs, routing, top=None):
    """-> (LEF paths, notes). routing: routing layer names, lowest first; top: highest signal routing layer."""
    if not routing:
        return list(lefs), []
    k = routing.index(top) + 1 if top in routing else len(routing)
    ok, above = set(routing[:k]), set(routing[k:])
    lay = {'low': routing[0], 'high': routing[k - 1]}
    size = {}
    for key, name in lay.items():
        w, p, s = _layer_size(lefs, name)
        size[key] = (max(2 * w, p), s)
    grid = _grid(lefs)
    dn = lambda v: math.floor(v / grid + 1e-6) * grid
    up = lambda v: math.ceil(v / grid - 1e-6) * grid
    dec = max(0, -int(math.floor(math.log10(grid)))) + 1
    out, fixed = [], []
    pin_re = re.compile(r'^[ \t]*PIN[ \t]+(\S+)[ \t]*\r?\n(.*?)^[ \t]*END[ \t]+\1\b', re.M | re.S)
    port_re = re.compile(r'^([ \t]*)PORT\b[^\n]*\n(.*?)^[ \t]*END\b[ \t]*\r?$', re.M | re.S)
    for l in lefs:
        t = Path(l).read_text(errors='replace')
        if not re.search(r'^\s*MACRO\s', t, re.M):
            out.append(l); continue
        changed = [False]

        def macro(mm):
            name, mbody = mm.group(1), mm.group(2)
            pins = list(pin_re.finditer(mbody))
            need = {}   # pin name -> {'low': [rects], 'high': [rects]}
            for pm in pins:
                body = pm.group(2)
                if re.search(r'^\s*USE\s+(POWER|GROUND)\b', body, re.M) or not port_re.search(body):
                    continue
                sh = _shapes(body)
                if not sh or any(lr in ok for lr, _ in sh):
                    continue
                need[pm.group(1)] = {'low': [r for lr, r in sh if lr not in above],
                                     'high': [r for lr, r in sh if lr in above]}
            if not need:
                return mm.group(0)
            # what each pad must keep clear of: every other pin's shapes on that layer (and the cell's OBS there)
            obs = re.search(r'^[ \t]*OBS\b(.*?)^[ \t]*END\b', mbody, re.M | re.S)
            obs_sh = _shapes(obs.group(1)) if obs else []
            pads = {p: {'low': [], 'high': []} for p in need}
            for key in ('low', 'high'):
                layer = lay[key]; side, space = size[key]
                placed = {p: [r for r in need[p][key]] for p in need}   # start from the pin outlines
                for pm in pins:
                    p = pm.group(1)
                    if p not in need:
                        continue
                    others = [r for q, rs in placed.items() if q != p for r in rs]
                    others += [r for pm2 in pins if pm2.group(1) != p for lr, r in _shapes(pm2.group(2)) if lr == layer]
                    others += [r for lr, r in obs_sh if lr == layer]
                    grown = [_grow(r, side, others, space) for r in need[p][key]]
                    placed[p] = grown
                    pads[p][key] = grown

            def pin(pm):
                p = pm.group(1)
                if p not in pads:
                    return pm.group(0)
                body = pm.group(2)
                port = port_re.search(body)
                add = ''
                for key in ('low', 'high'):
                    if pads[p][key]:
                        add += f'{port.group(1)}  LAYER {lay[key]} ;\n' + ''.join(
                            '        RECT ' + ' '.join('%.*f' % (dec, v) for v in (dn(r[0]), dn(r[1]), up(r[2]), up(r[3])))
                            + ' ;\n' for r in pads[p][key])
                changed[0] = True; fixed.append(f'{name}/{p}')
                return pm.group(0).replace(body, body[:port.end(2)] + add + body[port.end(2):], 1)
            nb = pin_re.sub(pin, mbody)
            return mm.group(0).replace(mbody, nb, 1)
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
        notes.append(f'{len(fixed)} signal pin(s) have no shape on the routing layers {routing[0]}-{routing[k - 1]} '
                     f'(e.g. {", ".join(fixed[:3])}): a shape is added at the same place (copy of the cell LEF), on '
                     f'{lay["low"]} under device-layer shapes, on {lay["high"]} under shapes above {lay["high"]}')
    return out, notes
