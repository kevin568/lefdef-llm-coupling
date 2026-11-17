#!/usr/bin/env python3
"""flow/lite.py: find the routed nets with the most coupling in a DEF and suggest lower-coupling routes for them.

Pure Python standard library (Python 3.6+). No OpenROAD, no GPU, no network: nothing leaves the machine.

    python3 flow/lite.py A.def B.lef [more.lef | lef_folder ...] [--spef A.spef] [--nets REGEX] [--top 20]

What it does
  1. Reads the LEF (routing layers, pitches, widths, vias, cell pins) and the routed DEF (wires of every net).
  2. Coupling of each signal net: the length (um) it runs parallel to another net's wire on the same layer within
     2 routing pitches (the geometric proxy used by the full lefdef-llm-coupling flow). With --spef (e.g. from
     StarRC, Calibre xRC, OpenRCX), the real coupling capacitance (fF) of each net is reported and used for ranking.
  3. For the top nets (2..8 pins), about 18 alternative trees per net (combs, MST L-shapes, Hanan trees) are scored
     with the full flow's heuristic: coupling that the new tree leaves behind minus new same-layer parallel length.
     The best tree, its predicted coupling reduction and wire-length change are reported.

What it does not do: it does not reroute, check DRC, or re-extract. The ranking of nets by coupling is reliable
(Spearman 0.90-0.96 against OpenRCX coupling on 6 public layouts). The predicted reduction of a suggested tree is an
optimistic geometric estimate: a real router does not follow the tree exactly and picks its own layers. On tempsense,
estimated -40% (median of 8 nets) against -0.2% measured after OpenROAD reroute + OpenRCX extraction (4 of 8 nets
lower, best -69%). Use the suggestions as a short list to reroute and re-extract, not as results.

Output (folder --out, default out/lite/<DEF name>): report.md, nets.csv, suggestions.csv, suggestions.json,
guides/<net>.txt (suggested tree segments in DEF units and um).
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

SIG = ('SIGNAL', 'ANALOG')
POWER = {'vdd', 'vss', 'gnd', 'vdda', 'vddd', 'vssa', 'vssd'}
K_PITCH = 2.0      # coupling window: wires within K routing pitches on the same layer
TOL_PITCH = 2.0    # a candidate span within TOL pitches of an existing track of the same net counts as kept


def die(msg):
    sys.stderr.write('ERROR: ' + msg + '\n')
    sys.exit(2)


def read_text(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', errors='replace') as f:
            return f.read()
    with open(path, errors='replace') as f:
        return f.read()


# ----------------------------------------------------------------------------------------------------------- LEF
class Lef(object):
    def __init__(self):
        self.layers = []          # routing layers in order: dict(name, pitch, width, dir) (microns)
        self.layer_by = {}
        self.vias = {}            # via name -> set of layer names
        self.viarules = set()
        self.macros = {}          # macro -> dict(size=(w,h), pins={pin: [(x1,y1,x2,y2)]})
        self.all_layers = []      # every LAYER (routing + cut + ...) in order

    def read(self, path):
        t = read_text(path)
        lines = t.splitlines()
        i, n = 0, len(lines)
        while i < n:
            ln = lines[i].strip()
            m = re.match(r'^LAYER\s+(\S+)\s*$', ln)
            if m:
                name = m.group(1); body = []; i += 1
                while i < n and not re.match(r'^\s*END\s+' + re.escape(name) + r'\b', lines[i]):
                    body.append(lines[i]); i += 1
                self._layer(name, '\n'.join(body))
            elif re.match(r'^VIA\s+\S+', ln) and not ln.endswith(';'):
                name = ln.split()[1]; layers = set(); i += 1
                while i < n and not re.match(r'^\s*END\s+' + re.escape(name) + r'\b', lines[i]):
                    mm = re.match(r'^\s*LAYER\s+(\S+)\s*;', lines[i])
                    if mm:
                        layers.add(mm.group(1))
                    i += 1
                self.vias[name] = layers
            elif re.match(r'^VIARULE\s+\S+', ln, re.I):
                self.viarules.add(ln.split()[1])
            elif re.match(r'^MACRO\s+\S+\s*$', ln):   # (PROPERTYDEFINITIONS has 'MACRO name TYPE ;' lines)
                name = ln.split()[1]; i += 1; mac = dict(size=(0., 0.), origin=(0., 0.), pins={})
                while i < n and not re.match(r'^\s*END\s+' + re.escape(name) + r'\s*$', lines[i]):
                    s = lines[i].strip()
                    mm = re.match(r'^SIZE\s+(\S+)\s+BY\s+(\S+)', s)
                    if mm:
                        mac['size'] = (float(mm.group(1)), float(mm.group(2)))
                    mm = re.match(r'^ORIGIN\s+(\S+)\s+(\S+)', s)
                    if mm:
                        mac['origin'] = (float(mm.group(1)), float(mm.group(2)))
                    mm = re.match(r'^PIN\s+(\S+)', s)
                    if mm:
                        pin = mm.group(1); rects = []; i += 1
                        while i < n and not re.match(r'^\s*END\s+' + re.escape(pin) + r'\s*$', lines[i]):
                            r = re.match(r'^\s*RECT\s+(?:MASK\s+\d+\s+)?(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', lines[i])
                            if r:
                                x1, y1, x2, y2 = map(float, r.groups())
                                rects.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                            p = re.match(r'^\s*POLYGON\s+(?:MASK\s+\d+\s+)?([-\d.\s]+)', lines[i])
                            if p:
                                v = [float(x) for x in p.group(1).split()]
                                xs, ys = v[0::2], v[1::2]
                                if xs and ys:
                                    rects.append((min(xs), min(ys), max(xs), max(ys)))
                            i += 1
                        mac['pins'][pin] = rects
                    i += 1
                self.macros[name] = mac
            i += 1

    def _layer(self, name, body):
        self.all_layers.append(name)
        if not re.search(r'^\s*TYPE\s+ROUTING\b', body, re.M):
            return
        if name in self.layer_by:
            return
        d = re.search(r'^\s*DIRECTION\s+(\S+)', body, re.M)
        p = re.search(r'^\s*PITCH\s+([\d.]+)(?:\s+([\d.]+))?', body, re.M)
        w = re.search(r'^\s*WIDTH\s+([\d.]+)\s*;', body, re.M)   # not the WIDTH rows of a SPACINGTABLE
        direction = d.group(1).upper() if d else 'HORIZONTAL'
        if p and p.group(2):   # PITCH x y: horizontal wires repeat in y
            pitch = float(p.group(2)) if direction.startswith('H') else float(p.group(1))
        else:
            pitch = float(p.group(1)) if p else 0.
        L = dict(name=name, pitch=pitch, width=float(w.group(1)) if w else 0., dir=direction, level=len(self.layers) + 1)
        self.layers.append(L); self.layer_by[name] = L


def lef_paths(items):
    out = []
    for it in items:
        for s in it.split(','):
            if not s:
                continue
            if os.path.isdir(s):
                found = sorted(os.path.join(s, f) for f in os.listdir(s) if f.endswith('.lef') or f.endswith('.lef.gz'))
                if not found:
                    die('no *.lef file in folder ' + s)
                out += found
            elif os.path.exists(s):
                out.append(s)
            else:
                die('missing file ' + s)
    return out


# ----------------------------------------------------------------------------------------------------------- DEF
TOK = re.compile(r'\(|\)|;|[^\s();]+')


def orient_pt(x, y, w, h, o):
    """cell-local point -> offset from the placed lower-left corner (LEF/DEF orientation rules)"""
    return {'N': (x, y), 'S': (w - x, h - y), 'W': (h - y, x), 'E': (y, w - x),
            'FN': (w - x, y), 'FS': (x, h - y), 'FW': (y, x), 'FE': (h - y, w - x)}.get(o, (x, y))


def orient_off(a, b, o):
    """pin shape offset (relative to the pin placement point) under orientation o"""
    return {'N': (a, b), 'S': (-a, -b), 'W': (-b, a), 'E': (b, -a),
            'FN': (-a, b), 'FS': (a, -b), 'FW': (b, a), 'FE': (-b, -a)}.get(o, (a, b))


def section(t, name):
    m = re.search(r'^\s*' + name + r'\s+\d+\s*;', t, re.M)
    if not m:
        return ''
    e = re.search(r'^\s*END\s+' + name + r'\b', t[m.end():], re.M)
    return t[m.end():m.end() + e.start()] if e else t[m.end():]


def statements(body):
    """'- name ... ;' statements of a DEF section"""
    out = []
    for st in re.split(r';\s*\n', body):
        st = st.strip()
        if st.startswith('-'):
            out.append(st)
    return out


class Def(object):
    def __init__(self, path, lef):
        t = read_text(path)
        m = re.search(r'^\s*DESIGN\s+(\S+)', t, re.M); self.design = m.group(1) if m else os.path.basename(path)
        m = re.search(r'^\s*UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', t, re.M); self.dbu = int(m.group(1)) if m else 1000
        self.lef = lef
        # vias defined in the DEF: name -> layers
        self.vias = dict(lef.vias)
        self.viarules_used = set()
        for st in statements(section(t, 'VIAS')):
            tk = st.split()
            name = tk[1]; layers = set()
            if '+ LAYERS' in st or '+LAYERS' in st:
                mm = re.search(r'\+\s*LAYERS\s+(\S+)\s+(\S+)\s+(\S+)', st); layers = {mm.group(1), mm.group(3)} if mm else set()
            layers |= set(re.findall(r'\+\s*RECT\s+(\S+)', st)) | set(re.findall(r'\+\s*POLYGON\s+(\S+)', st))
            mm = re.search(r'\+\s*VIARULE\s+(\S+)', st)
            if mm:
                self.viarules_used.add(mm.group(1))
            self.vias[name] = layers
        self.has_tracks = bool(re.search(r'^\s*TRACKS\s', t, re.M))
        # components
        self.comps = {}
        for st in statements(section(t, 'COMPONENTS')):
            tk = TOK.findall(st)
            inst, mac = tk[1], tk[2]
            loc = None
            for j, x in enumerate(tk):
                if x in ('PLACED', 'FIXED', 'COVER') and j + 5 < len(tk) and tk[j + 1] == '(':
                    loc = (int(float(tk[j + 2])), int(float(tk[j + 3])), tk[j + 5]); break
            self.comps[inst] = (mac, loc)
        # I/O pins
        self.pins = {}
        for st in statements(section(t, 'PINS')):
            tk = TOK.findall(st)
            name = tk[1]; rect = None; loc = None
            for j, x in enumerate(tk):
                if x == 'LAYER' and j + 10 < len(tk) and rect is None:
                    nums = [v for v in tk[j + 2:j + 14] if re.match(r'^-?\d+(\.\d+)?$', v)]
                    if len(nums) >= 4:
                        rect = tuple(int(float(v)) for v in nums[:4])
                if x in ('PLACED', 'FIXED', 'COVER') and j + 5 < len(tk) and tk[j + 1] == '(' and loc is None:
                    loc = (int(float(tk[j + 2])), int(float(tk[j + 3])), tk[j + 5])
            if rect and loc:
                cx, cy = (rect[0] + rect[2]) / 2., (rect[1] + rect[3]) / 2.
                dx, dy = orient_off(cx, cy, loc[2])
                self.pins[name] = (int(loc[0] + dx), int(loc[1] + dy))
        # nets
        self.nets = {}
        self.unknown_vias = set()
        for st in statements(section(t, 'NETS')):
            self._net(st)

    def _layer_w(self, name):
        L = self.lef.layer_by.get(name)
        return int(round(L['width'] * self.dbu)) if L else 0

    def _pin_xy(self, inst, pin):
        if inst == 'PIN':
            return self.pins.get(pin)
        mac, loc = self.comps.get(inst, (None, None))
        if not loc or mac not in self.lef.macros:
            return None
        M = self.lef.macros[mac]
        rects = M['pins'].get(pin) or M['pins'].get(pin.replace('\\', ''))
        if not rects:
            return None
        w, h = M['size']; ox, oy = M['origin']
        cx = sum((r[0] + r[2]) / 2. for r in rects) / len(rects) + ox
        cy = sum((r[1] + r[3]) / 2. for r in rects) / len(rects) + oy
        dx, dy = orient_pt(cx, cy, w, h, loc[2])
        return (int(round(loc[0] + dx * self.dbu)), int(round(loc[1] + dy * self.dbu)))

    def _net(self, st):
        tk = TOK.findall(st)
        name = tk[1]; i = 2; conns = []
        while i < len(tk) and tk[i] == '(':
            j = tk.index(')', i)
            if j - i >= 3:
                conns.append((tk[i + 1], tk[i + 2]))
            i = j + 1
        use = 'SIGNAL'; shapes = []; vias = 0; routed = False
        layer = None; prev = None; routing = False; width = 0
        while i < len(tk):
            x = tk[i]
            if x == '+':
                kw = tk[i + 1] if i + 1 < len(tk) else ''
                if kw in ('ROUTED', 'FIXED', 'COVER', 'NOSHIELD'):
                    routing = True; routed = True; layer = tk[i + 2]; width = self._layer_w(layer); prev = None; i += 3; continue
                if routing and kw in ('SHAPE', 'MASK'):
                    i += 3; continue
                routing = False
                if kw == 'USE':
                    use = tk[i + 2].upper()
                i += 2; continue
            if not routing:
                i += 1; continue
            if x == 'NEW':
                layer = tk[i + 1]; width = self._layer_w(layer); prev = None; i += 2; continue
            if x == '(':
                j = tk.index(')', i); v = tk[i + 1:j]
                px = prev[0] if v[0] == '*' and prev else int(float(v[0]))
                py = prev[1] if len(v) > 1 and v[1] == '*' and prev else int(float(v[1]))
                ext = int(float(v[2])) if len(v) > 2 and v[2] != '*' else None
                cur = (px, py, ext)
                if prev and (prev[0], prev[1]) != (px, py) and width:
                    e1 = prev[2] if prev[2] is not None else width // 2
                    e2 = ext if ext is not None else width // 2
                    hw = width // 2
                    if prev[1] == py:   # horizontal
                        a, b = (prev, cur) if prev[0] < px else (cur, prev)
                        ea = e1 if a is prev else e2; eb = e2 if b is cur else e1
                        shapes.append((layer, (a[0] - ea, py - hw, b[0] + eb, py + hw)))
                    elif prev[0] == px:
                        a, b = (prev, cur) if prev[1] < py else (cur, prev)
                        ea = e1 if a is prev else e2; eb = e2 if b is cur else e1
                        shapes.append((layer, (px - hw, a[1] - ea, px + hw, b[1] + eb)))
                prev = cur; i = j + 1; continue
            if x == 'RECT':   # small patch rectangle: not a wire (OpenDB's wire path does not list it either)
                i = tk.index(')', i) + 1; continue
            if x == 'VIRTUAL':
                j = tk.index(')', i); v = tk[i + 2:j]
                prev = (int(float(v[0])), int(float(v[1])), None); i = j + 1; continue
            if x in ('MASK', 'STYLE', 'TAPERRULE'):
                i += 2; continue
            if x == 'TAPER':
                i += 1; continue
            if x in ('N', 'S', 'E', 'W', 'FN', 'FS', 'FE', 'FW'):
                i += 1; continue
            if re.match(r'^-?\d+$', x):   # e.g. a width after the layer
                i += 1; continue
            # a via: the path continues on the via's other routing layer
            vias += 1
            vl = [l for l in self.vias.get(x, ()) if l in self.lef.layer_by]
            if not vl:
                self.unknown_vias.add(x)
            other = [l for l in vl if l != layer]
            if other:
                layer = other[0]; width = self._layer_w(layer)
            i += 1
        pins = [p for p in (self._pin_xy(a, b) for a, b in conns) if p]
        self.nets[name] = dict(use=use, shapes=shapes, vias=vias, routed=routed and bool(shapes), conns=len(conns), pins=pins)


# ------------------------------------------------------------------------------------- coupling proxy (full flow)
class Proxy(object):
    """Same rule as the full flow's proxy: same-layer parallel runs within K pitches, runs >= 1 um, aspect >= 3."""

    def __init__(self, lef, dbu):
        self.dbu = dbu
        self.pitch = {L['name']: L['pitch'] for L in lef.layers}
        self.min_len = dbu

    def runs(self, shapes):
        groups = defaultdict(list)
        for layer, r in shapes:
            if layer not in self.pitch:
                continue
            w, h = r[2] - r[0], r[3] - r[1]
            if w >= h:
                if w < self.min_len or w < 3.0 * max(h, 1):
                    continue
                groups[(layer, 'H', r[1], r[3])].append((r[0], r[2]))
            else:
                if h < self.min_len or h < 3.0 * max(w, 1):
                    continue
                groups[(layer, 'V', r[0], r[2])].append((r[1], r[3]))
        out = defaultdict(list)
        for (layer, o, lo, hi), iv in groups.items():
            iv.sort(); cur = list(iv[0])
            for a, b in iv[1:]:
                if a <= cur[1]:
                    cur[1] = max(cur[1], b)
                else:
                    out[(layer, o)].append((lo, hi, cur[0], cur[1])); cur = [a, b]
            out[(layer, o)].append((lo, hi, cur[0], cur[1]))
        return out

    def gmax(self, layer):
        return K_PITCH * self.pitch[layer] * self.dbu

    def coupling(self, nets):
        """{net: um}, {(n1, n2): um}"""
        by = defaultdict(list)
        for net, shapes in nets.items():
            for key, rr in self.runs(shapes).items():
                for lo, hi, a, b in rr:
                    by[key].append((lo, hi, a, b, net))
        per = defaultdict(int); pair = defaultdict(int)
        for (layer, o), rr in by.items():
            g = self.gmax(layer); rr.sort()
            for i, (lo1, hi1, a1, b1, n1) in enumerate(rr):
                for lo2, hi2, a2, b2, n2 in rr[i + 1:]:
                    if lo2 - hi1 > g:
                        break
                    if n1 == n2:
                        continue
                    gap = max(lo2 - hi1, lo1 - hi2)
                    if gap <= 0 or gap > g:
                        continue
                    ov = min(b1, b2) - max(a1, a2)
                    if ov <= 0:
                        continue
                    per[n1] += ov; per[n2] += ov; pair[tuple(sorted((n1, n2)))] += ov
        return {k: v / float(self.dbu) for k, v in per.items()}, {k: v / float(self.dbu) for k, v in pair.items()}


# --------------------------------------------------------------------------------- candidate trees (full flow recipe)
def merge(segs):
    groups = defaultdict(list)
    for x1, y1, x2, y2 in segs:
        if (x1, y1) == (x2, y2):
            continue
        if y1 == y2:
            groups[('H', y1)].append(sorted([x1, x2]))
        else:
            groups[('V', x1)].append(sorted([y1, y2]))
    out = []
    for (o, t), ivs in sorted(groups.items()):
        m = []
        for a, b in sorted(ivs):
            if m and a <= m[-1][1]:
                m[-1][1] = max(b, m[-1][1])
            else:
                m.append([a, b])
        out += [[a, t, b, t] if o == 'H' else [t, a, t, b] for a, b in m]
    return out


def lshape(p, q, hfirst):
    bend = (q[0], p[1]) if hfirst else (p[0], q[1])
    return [[p[0], p[1], bend[0], bend[1]], [bend[0], bend[1], q[0], q[1]]]


def covers(geom, pins):
    return all(any((y == y1 == y2 and min(x1, x2) <= x <= max(x1, x2)) or (x == x1 == x2 and min(y1, y2) <= y <= max(y1, y2))
                   for x1, y1, x2, y2 in geom) for x, y in pins)


def mst_edges(pins):
    n = len(pins); used = {0}; edges = []
    while len(used) < n:
        d, a, b = min((abs(pins[i][0] - pins[j][0]) + abs(pins[i][1] - pins[j][1]), i, j)
                      for i in used for j in range(n) if j not in used)
        edges.append((a, b)); used.add(b)
    return edges


def hanan(points, rng):
    xs = sorted({p[0] for p in points}); ys = sorted({p[1] for p in points}); coords = [(x, y) for x in xs for y in ys]
    ids = {p: i for i, p in enumerate(coords)}; ny = len(ys); edges = []; f = math.exp(rng.uniform(-.7, .7))
    for i, (x, y) in enumerate(coords):
        for j in ([i + ny] if i + ny < len(coords) else []) + ([i + 1] if i % ny + 1 < ny else []):
            X, Y = coords[j]
            edges.append(((abs(X - x) * f + abs(Y - y) / f) * math.exp(rng.uniform(-1.4, 1.4)), i, j))
    parent = list(range(len(coords)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    adj = [set() for _ in coords]
    for _, a, b in sorted(edges):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb; adj[a].add(b); adj[b].add(a)
    term = {ids[tuple(p)] for p in points}
    todo = [i for i, a in enumerate(adj) if len(a) == 1 and i not in term]; removed = set()
    while todo:
        a = todo.pop()
        if a in removed:
            continue
        removed.add(a)
        for b in list(adj[a]):
            adj[b].remove(a)
            if len(adj[b]) == 1 and b not in term:
                todo.append(b)
        adj[a].clear()
    return merge([[coords[a][0], coords[a][1], coords[b][0], coords[b][1]] for a in range(len(coords)) for b in adj[a] if a < b])


def pool_for_net(net, pins):
    pins = sorted(set(tuple(p) for p in pins))
    rng = random.Random(int(hashlib.sha256(('lite:' + net).encode()).hexdigest()[:8], 16))
    opts = [('x_first', merge([s for p in pins[1:] for s in lshape(pins[0], p, True)])),
            ('y_first', merge([s for p in pins[1:] for s in lshape(pins[0], p, False)]))]
    edges = mst_edges(pins)
    for t in range(8):
        dirs = [rng.random() < .5 for _ in edges]
        opts.append(('mst%d' % t, merge([s for k, (a, b) in enumerate(edges) for s in lshape(pins[a], pins[b], dirs[k])])))
    for t in range(8):
        opts.append(('hanan%d' % t, hanan(pins, rng)))
    seen = set(); out = []
    for kind, g in opts:
        key = tuple(map(tuple, g))
        if key in seen or not covers(g, pins):
            continue
        seen.add(key); out.append((kind, g))
    return out


# --------------------------------------------------------------------------------- heuristic score (full flow rule)
def union_len(iv):
    iv = sorted(iv); tot = 0; cur = None
    for a, b in iv:
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        tot += cur[1] - cur[0]
    return tot


def covered(a, b, cover):
    return union_len([(max(a, x), min(b, y)) for x, y in cover if min(b, y) > max(a, x)])


def segs(geometry):
    out = []
    for x1, y1, x2, y2 in geometry:
        if y1 == y2 and x1 != x2:
            out.append(('H', y1, min(x1, x2), max(x1, x2)))
        elif x1 == x2 and y1 != y2:
            out.append(('V', x1, min(y1, y2), max(y1, y2)))
    return out


def analyse(P, net, nets, geometry):
    """removed coupling (um), new same-layer parallel length (um), new path length (um), tree length (um)"""
    tolf = lambda layer: TOL_PITCH * P.pitch[layer] * P.dbu
    orun = P.runs(nets.get(net, []))
    cs = segs(geometry)
    other_runs = {o: P.runs(s) for o, s in nets.items() if o != net}
    used = {o: {l for (l, oo) in orun if oo == o} for o in 'HV'}
    removed = new_same = new_len = 0.
    near = lambda o, lo, hi, layer: [(a, b) for oo, t, a, b in cs if oo == o and lo - tolf(layer) <= t <= hi + tolf(layer)]
    for o_net, orr in other_runs.items():
        for key, rr in orun.items():
            layer, o = key; g = P.gmax(layer)
            for lo1, hi1, a1, b1 in rr:
                for lo2, hi2, a2, b2 in orr.get(key, []):
                    gap = max(lo2 - hi1, lo1 - hi2)
                    if gap <= 0 or gap > g:
                        continue
                    a, b = max(a1, a2), min(b1, b2)
                    if b > a:
                        removed += (b - a) - covered(a, b, near(o, lo1, hi1, layer))
    for o, t, a, b in cs:
        cover = [(ra, rb) for (layer, oo), rr in orun.items() if oo == o for lo, hi, ra, rb in rr
                 if lo - tolf(layer) <= t <= hi + tolf(layer)]
        fresh = [(a, b)]
        for x, y in sorted(cover):
            nxt = []
            for p, q in fresh:
                if y <= p or x >= q:
                    nxt.append((p, q)); continue
                if x > p:
                    nxt.append((p, x))
                if y < q:
                    nxt.append((y, q))
            fresh = nxt
        for p, q in fresh:
            new_len += q - p
            for o_net, orr in other_runs.items():
                for (layer, oo), rr in orr.items():
                    if oo != o:
                        continue
                    g = P.gmax(layer)
                    for lo, hi, ra, rb in rr:
                        d = max(lo - t, t - hi)
                        if d > g or d < -g:
                            continue
                        ov = min(q, rb) - max(p, ra)
                        if ov > 0 and (layer in used[o] or not used[o]):
                            new_same += ov
    tree = sum(b - a for _, _, a, b in cs)
    return removed / P.dbu, new_same / P.dbu, new_len / P.dbu, tree / P.dbu


def window(nets, rects, margin):
    x0 = min(r[0] for r in rects) - margin; y0 = min(r[1] for r in rects) - margin
    x1 = max(r[2] for r in rects) + margin; y1 = max(r[3] for r in rects) + margin
    return {n: s for n, s in nets.items() if any(r[0] <= x1 and r[2] >= x0 and r[1] <= y1 and r[3] >= y0 for _, r in s)}


# ----------------------------------------------------------------------------------------------------------- SPEF
def read_spef(path):
    """per-net coupling (fF), each coupling capacitor credited to both nets"""
    names, unit, cur, sec = {}, 1e3, None, None
    pairs, pin_owner = {}, {}
    for ln in read_text(path).splitlines():
        if ln.startswith('*C_UNIT'):
            v, u = ln.split()[1:3]; unit = float(v) * {'PF': 1e3, 'FF': 1.0}.get(u.upper(), 1e3)
        elif cur is None and ln.startswith('*') and len(ln.split()) == 2 and ln.split()[0][1:].isdigit():
            k, v = ln.split(); names[k] = v
        elif ln.startswith('*D_NET'):
            k = ln.split()[1]; cur = names.get(k, k); sec = None
        elif ln.startswith('*CONN'):
            sec = 'conn'
        elif ln.startswith('*CAP'):
            sec = 'cap'
        elif ln.startswith('*RES') or ln.startswith('*END'):
            sec = None
        elif sec == 'conn' and (ln.startswith('*I') or ln.startswith('*P')):
            pin_owner[ln.split()[1]] = cur
        elif sec == 'cap':
            f = ln.split()
            if len(f) == 4:
                pairs[frozenset([f[1], f[2]])] = float(f[3]) * unit

    def owner(node):
        if node in pin_owner:
            return pin_owner[node]
        k = node.split(':')[0]
        return names.get(k, k)
    per = defaultdict(float)
    for key, v in pairs.items():
        ab = list(key)
        if len(ab) == 2:
            for n in {owner(ab[0]), owner(ab[1])}:
                per[n.replace('\\', '')] += v
    return dict(per)


# ----------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 usage='python3 flow/lite.py A.def B.lef [more LEF files/folders] [options]')
    ap.add_argument('inputs', nargs='+', help='routed DEF first, then LEF files or folders (.gz accepted)')
    ap.add_argument('--spef', help='SPEF of this layout (StarRC / Calibre / OpenRCX): rank by real coupling (fF)')
    ap.add_argument('--nets', help='regular expression: only suggest routes for nets whose name matches')
    ap.add_argument('--top', type=int, default=20, help='how many top-coupling nets get route suggestions (default 20)')
    ap.add_argument('--out', help='output folder (default out/lite/<DEF name>)')
    a = ap.parse_args()
    deff = a.inputs[0]
    if not re.search(r'\.def(\.gz)?$', deff, re.I):
        die('give the DEF first: python3 flow/lite.py A.def B.lef')
    if len(a.inputs) < 2:
        die('give the LEF files after the DEF: python3 flow/lite.py A.def B.lef')
    if not os.path.exists(deff):
        die('missing file ' + deff)
    lef = Lef()
    for p in lef_paths(a.inputs[1:]):
        lef.read(p)
    if not lef.layers:
        die('no routing LAYER in the LEF files: the technology LEF must be given')
    print('LEF: %d routing layers (%s), %d macros' % (len(lef.layers), ', '.join(L['name'] for L in lef.layers), len(lef.macros)))
    D = Def(deff, lef)
    missing = sorted({m for m, _ in D.comps.values()} - set(lef.macros))
    if missing:
        die('%d cell/macro type(s) used in the DEF are not in the LEF files, e.g. %s. Add the LEF that defines them.'
            % (len(missing), ', '.join(missing[:8])))
    bad_rules = sorted(D.viarules_used - lef.viarules)
    if bad_rules:
        print('  note: the DEF uses via rule(s) %s not in the LEF (a different PDK version?); wires are still read'
              % ', '.join(bad_rules))
    if D.unknown_vias:
        print('  note: %d via name(s) not defined in the LEF or DEF (e.g. %s); layer changes at them may be missed'
              % (len(D.unknown_vias), sorted(D.unknown_vias)[0]))
    sig = {n: v for n, v in D.nets.items() if v['use'] in SIG and v['routed'] and n.lower() not in POWER}
    print('DEF %s: %d nets, %d routed signal nets, %d components, DBU %d'
          % (D.design, len(D.nets), len(sig), len(D.comps), D.dbu))
    if not sig:
        die('no routed signal net in the DEF')
    P = Proxy(lef, D.dbu)
    nets = {n: v['shapes'] for n, v in sig.items()}
    per, pair = P.coupling(nets)
    spef = read_spef(a.spef) if a.spef else None
    plain = lambda s: s.replace('\\', '')
    metric = (lambda n: spef.get(plain(n), 0.)) if spef else (lambda n: per.get(n, 0.))
    unit = 'fF (SPEF)' if spef else 'um parallel length (proxy)'
    wl = lambda shapes: sum(max(r[2] - r[0], r[3] - r[1]) - min(r[2] - r[0], r[3] - r[1]) for _, r in shapes) / float(D.dbu)
    nbr = defaultdict(list)
    for (n1, n2), um in pair.items():
        nbr[n1].append((n2, um)); nbr[n2].append((n1, um))
    rows = []
    for n, v in sig.items():
        nb = sorted(nbr.get(n, []), key=lambda x: -x[1])
        rows.append(dict(net=n, coupling=metric(n), proxy_um=per.get(n, 0.), spef_fF=spef.get(plain(n)) if spef else None,
                         wl_um=wl(v['shapes']), vias=v['vias'], pins=v['conns'],
                         top_neighbors='; '.join('%s %.1f' % (m, um) for m, um in nb[:3])))
    rows.sort(key=lambda r: (-r['coupling'], r['net']))
    out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out', 'lite',
                                 re.sub(r'\.def(\.gz)?$', '', os.path.basename(deff), flags=re.I))
    if not os.path.isdir(os.path.join(out, 'guides')):
        os.makedirs(os.path.join(out, 'guides'))
    with open(os.path.join(out, 'nets.csv'), 'w') as f:
        w = csv.writer(f); w.writerow(['rank', 'net', 'coupling (' + unit + ')', 'proxy_um', 'spef_fF', 'wl_um', 'vias', 'pins', 'top neighbors (um)'])
        for i, r in enumerate(rows):
            w.writerow([i + 1, r['net'], round(r['coupling'], 4), round(r['proxy_um'], 2),
                        '' if r['spef_fF'] is None else round(r['spef_fF'], 4), round(r['wl_um'], 2), r['vias'], r['pins'], r['top_neighbors']])
    # route suggestions
    rx = re.compile(a.nets) if a.nets else None
    ok = lambda r, hi: (2 <= r['pins'] <= hi and len(D.nets[r['net']]['pins']) == r['pins']
                        and (not rx or rx.search(r['net'])) and r['proxy_um'] > 0)
    pool = [r for r in rows if ok(r, 8)][:a.top]
    if not pool:   # analog layouts (multi-finger devices): every net has more than 8 pins; take them all
        pool = [r for r in rows if ok(r, 10 ** 6)][:a.top]
        if pool:
            print('no net with 2..8 pins: route suggestions for the top %d nets with any pin count' % len(pool))
    sugg = []
    margin = 10 * D.dbu
    for k, r in enumerate(pool):
        n = r['net']; pins = D.nets[n]['pins']
        best = None
        for kind, g in pool_for_net(n, pins):
            rects = [[min(s[0], s[2]), min(s[1], s[3]), max(s[0], s[2]), max(s[1], s[3])] for s in g] + [list(rr) for _, rr in nets[n]]
            win = window(nets, rects, margin); win[n] = nets[n]
            rem, new_same, new_len, tree = analyse(P, n, win, g)
            h = rem - new_same
            if best is None or h > best['score']:
                best = dict(kind=kind, geometry=g, score=h, removed_um=rem, new_same_layer_um=new_same, new_path_um=new_len, tree_um=tree)
        if best is None:
            continue
        pred = min(best['score'], r['proxy_um'])
        sugg.append(dict(net=n, proxy_um=r['proxy_um'], spef_fF=r['spef_fF'], wl_um=r['wl_um'], best_tree=best['kind'],
                         estimated_coupling_removed_um=round(pred, 3),
                         estimated_coupling_change_pct=round(-100. * pred / r['proxy_um'], 1) if r['proxy_um'] else 0.,
                         tree_wl_um=round(best['tree_um'], 3), wl_change_um=round(best['tree_um'] - r['wl_um'], 3),
                         removed_um=round(best['removed_um'], 3), new_same_layer_um=round(best['new_same_layer_um'], 3),
                         geometry_dbu=best['geometry']))
        with open(os.path.join(out, 'guides', re.sub(r'[^\w.-]', '_', n) + '.txt'), 'w') as f:
            f.write('# net %s  suggested tree %s  (x1 y1 x2 y2, DEF units; DBU %d per um)\n' % (n, best['kind'], D.dbu))
            for s in best['geometry']:
                f.write('%d %d %d %d    # um: %.3f %.3f %.3f %.3f\n' % tuple(list(s) + [v / float(D.dbu) for v in s]))
        sys.stdout.write('\r  scored %d/%d nets' % (k + 1, len(pool))); sys.stdout.flush()
    print('')
    sugg.sort(key=lambda s: -s['estimated_coupling_removed_um'])
    with open(os.path.join(out, 'suggestions.json'), 'w') as f:
        json.dump(dict(def_file=os.path.abspath(deff), design=D.design, dbu=D.dbu, ranked_by=unit, suggestions=sugg), f, indent=1)
    with open(os.path.join(out, 'suggestions.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(['net', 'coupling now (um proxy)', 'spef_fF', 'best tree', 'estimated coupling removed (um, optimistic)', 'estimated change %',
                    'wire length now (um)', 'tree length (um)', 'WL change (um)'])  # estimates: optimistic
        for s in sugg:
            w.writerow([s['net'], round(s['proxy_um'], 2), '' if s['spef_fF'] is None else round(s['spef_fF'], 4), s['best_tree'],
                        s['estimated_coupling_removed_um'], s['estimated_coupling_change_pct'], round(s['wl_um'], 2), s['tree_wl_um'], s['wl_change_um']])
    good = [s for s in sugg if s['estimated_coupling_removed_um'] > 0]
    tot_now = sum(s['proxy_um'] for s in good); tot_rem = sum(s['estimated_coupling_removed_um'] for s in good)
    L = ['# Coupling scan: %s' % D.design, '',
         'DEF `%s`, %d routed signal nets. Ranked by %s. Coupling proxy: same-layer parallel length within %.0f routing '
         'pitches. No reroute, DRC or extraction was run: the net ranking is reliable, the reduction estimates are '
         'optimistic geometric upper bounds (confirm by rerouting and re-extracting).' % (os.path.basename(deff), len(sig), unit, K_PITCH), '',
         '## Key numbers', '',
         '| | value |', '|---|---:|',
         '| nets with a lower-coupling tree (geometric estimate) | %d of %d scored |' % (len(good), len(sugg)),
         '| their coupling now (proxy) | %.1f um |' % tot_now,
         '| estimated removable (proxy, optimistic upper bound) | %.1f um (%.1f%%) |' % (tot_rem, 100. * tot_rem / tot_now if tot_now else 0.),
         '| design total coupling (proxy, context) | %.1f um |' % (sum(per.values()) / 2.), '',
         '## Top nets by coupling', '', '| rank | net | coupling | WL (um) | pins | top neighbors (um) |', '|---:|---|---:|---:|---:|---|']
    for i, r in enumerate(rows[:min(len(rows), max(a.top, 20))]):
        L.append('| %d | `%s` | %.3f | %.1f | %d | %s |' % (i + 1, r['net'], r['coupling'], r['wl_um'], r['pins'], r['top_neighbors']))
    L += ['', '## Suggested reroutes (geometric estimate, optimistic: reroute and re-extract to confirm)', '',
          '| net | coupling now (um) | best tree | estimated removed (um) | est. change | WL now -> tree (um) | guide |',
          '|---|---:|---|---:|---:|---|---|']
    for s in sugg:
        L.append('| `%s` | %.1f | %s | %.1f | %+.1f%% | %.1f -> %.1f | guides/%s.txt |'
                 % (s['net'], s['proxy_um'], s['best_tree'], s['estimated_coupling_removed_um'], s['estimated_coupling_change_pct'],
                    s['wl_um'], s['tree_wl_um'], re.sub(r'[^\w.-]', '_', s['net'])))
    with open(os.path.join(out, 'report.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('')
    print('top nets by coupling (%s):' % unit)
    for r in rows[:5]:
        print('  %-40s %10.3f' % (r['net'][:40], r['coupling']))
    print('route suggestions: %d of %d scored nets have a lower-coupling tree; estimated %.1f of %.1f um removable '
          '(%.1f%%, proxy; optimistic upper bound, confirm by reroute + extraction)'
          % (len(good), len(sugg), tot_rem, tot_now, 100. * tot_rem / tot_now if tot_now else 0.))
    print('report: %s' % os.path.join(out, 'report.md'))


if __name__ == '__main__':
    main()
