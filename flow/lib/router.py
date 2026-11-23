#!/usr/bin/env python3
"""Built-in track router: a simplified detailed router in the spirit of MAGICAL's anaroute, used when OpenROAD's
detailed router cannot handle a technology (e.g. the 2024 conda OpenROAD on commercial FinFET PDKs).

What it does, per target net (the other nets are never touched):
  - reads the LEF (routing layers: direction, pitch, width, spacing; fixed vias; cell pins / OBS) and the DEF
    (components, I/O pins, every net's wires and vias, special nets);
  - builds a track lattice over a window around the net: every layer on its own tracks (LEF pitch; offset where the
    design's own wires on that layer run), nodes where tracks of the layers cross;
    a grid node is blocked when a shape of another net, another pin or an OBS lies within spacing + half width of it
    on that layer; a via is allowed only where its pads keep spacing too;
  - rips up the net and connects its pins with A* (preferred direction cost 1, wrong way 3, via 4; in guide mode
    +3 per node outside the candidate tree's corridor, so the route follows the tree the policy chose), growing a
    tree one pin at a time; pins are reached on the lowest signal layer with a via stack down to the pin layer;
  - writes the DEF with only that net's wiring replaced, route.log (with the GCELLGRID the flow reads), grt.guide
    (pin-access boxes) and drc.rpt from a spacing self-check of the new wires (an honest report: a reroute with a
    violation is not valid for the flow).
It is a simplified router: spacing / width / via-pad checks only, not a signoff DRC.

  python3 flow/lib/router.py --lefs a.lef,b.lef --def in.def --out out.def --nets "n1 n2" --layers M2-M4 \
      [--guide targets.guide] --log route.log --drc drc.rpt --guide-out grt.guide
"""
import argparse
import heapq
import re
import sys
import time
from pathlib import Path

import numpy as np

TOK = re.compile(r'\(|\)|;|[^\s();]+')
ORIENTS = ('N', 'S', 'E', 'W', 'FN', 'FS', 'FE', 'FW')


class NoPath(RuntimeError):
    """a net that cannot be connected; pin = (inst, pin) where it failed (None: the net as a whole)"""
    def __init__(self, msg, pin=None, also=None):
        RuntimeError.__init__(self, msg); self.pin = pin; self.also = also   # also: the other end (the pin the path started from)


def orient_pt(x, y, w, h, o):
    return {'N': (x, y), 'S': (w - x, h - y), 'W': (h - y, x), 'E': (y, w - x),
            'FN': (w - x, y), 'FS': (x, h - y), 'FW': (y, x), 'FE': (h - y, w - x)}.get(o, (x, y))


def orient_off(a, b, o):
    return {'N': (a, b), 'S': (-a, -b), 'W': (-b, a), 'E': (b, -a),
            'FN': (-a, b), 'FS': (a, -b), 'FW': (b, a), 'FE': (-b, -a)}.get(o, (a, b))


def poly_rects(v):
    """rectangles covering a polygon given as flat [x1, y1, x2, y2, ...] (rectilinear: exact; slabs between the
    vertex y values, inside by the even-odd rule at the slab middle); not rectilinear: its bounding box"""
    P = list(zip(v[0::2], v[1::2]))
    if len(P) >= 2 and P[0] == P[-1]:
        P = P[:-1]
    E = list(zip(P, P[1:] + P[:1]))
    if len(P) < 4 or any(a[0] != b[0] and a[1] != b[1] for a, b in E):
        return [(min(v[0::2]), min(v[1::2]), max(v[0::2]), max(v[1::2]))]
    ys = sorted({p[1] for p in P}); out = []
    for y0, y1 in zip(ys, ys[1:]):
        ym = (y0 + y1) / 2.
        xs = sorted(a[0] for a, b in E if a[0] == b[0] and min(a[1], b[1]) < ym < max(a[1], b[1]))
        out += [(xs[k], y0, xs[k + 1], y1) for k in range(0, len(xs) - 1, 2) if xs[k + 1] > xs[k]]
    return out or [(min(v[0::2]), min(v[1::2]), max(v[0::2]), max(v[1::2]))]


def _num(s, d=0.):
    try:
        return float(s)
    except (TypeError, ValueError):
        return d


# ---------------------------------------------------------------------------------------------------- LEF
class Tech(object):
    def __init__(self, lefs):
        self.order = []        # every layer in LEF order
        self.types = {}
        self.routing = {}      # name -> dict(dir, pitch, width, spacing) microns
        self.vias = {}         # fixed via name -> [(layer, (x1, y1, x2, y2))] microns
        self.macros = {}       # name -> dict(size, origin, pins {pin: [(layer, rect)]}, obs [(layer, rect)])
        self.mgrid = 0.
        for l in lefs:
            t = Path(l).read_text(errors='replace')
            m = re.search(r'^\s*MANUFACTURINGGRID\s+([\d.]+)', t, re.M)
            if m and not self.mgrid:
                self.mgrid = float(m.group(1))
            self._read(t)
        self.mgrid = self.mgrid or 0.001
        self.rorder = [n for n in self.order if n in self.routing]

    def _read(self, t):
        for m in re.finditer(r'^[ \t]*LAYER[ \t]+(\S+)[ \t]*\r?\n(.*?)^[ \t]*END[ \t]+\1\b', t, re.M | re.S):
            name, b = m.group(1), m.group(2)
            ty = re.search(r'^\s*TYPE\s+(\S+)', b, re.M)
            if not ty or name in self.types:
                continue
            self.order.append(name); self.types[name] = ty.group(1)
            if ty.group(1) != 'ROUTING':
                continue
            d = re.search(r'^\s*DIRECTION\s+(\S+)', b, re.M)
            p = re.search(r'^\s*PITCH\s+([\d.]+)(?:\s+([\d.]+))?', b, re.M)
            of = re.search(r'^\s*OFFSET\s+([\d.]+)(?:\s+([\d.]+))?', b, re.M)
            w = re.search(r'^\s*WIDTH\s+([\d.]+)\s*;', b, re.M)
            s = re.search(r'^\s*SPACING\s+([\d.]+)\s*;', b, re.M)
            st = re.search(r'SPACINGTABLE.*?WIDTH\s+[\d.]+((?:\s+[\d.]+)+)', b, re.S)
            dr = d.group(1).upper() if d else 'HORIZONTAL'
            pitch = (_num(p.group(2)) if dr.startswith('H') else _num(p.group(1))) if p and p.group(2) else (_num(p.group(1)) if p else 0.)
            width = _num(w.group(1)) if w else 0.
            spacing = _num(s.group(1)) if s else (_num(st.group(1).split()[0]) if st else 0.)
            if not spacing:
                spacing = max(pitch - width, width)
            off = ((_num(of.group(2)) if dr.startswith('H') else _num(of.group(1))) if of.group(2) else _num(of.group(1))) if of else None
            self.routing[name] = dict(dir=dr, pitch=pitch, width=width, spacing=spacing, offset=off)
        for m in re.finditer(r'^[ \t]*VIA[ \t]+(\S+)([^\n]*)\n(.*?)^[ \t]*END[ \t]+\1\b', t, re.M | re.S):
            if 'GENERATE' in m.group(2) or m.group(1) in self.vias:
                continue
            self.vias[m.group(1)] = self._shapes(m.group(3))
        for m in re.finditer(r'^[ \t]*MACRO[ \t]+(\S+)[ \t]*\r?\n(.*?)^[ \t]*END[ \t]+\1\b', t, re.M | re.S):
            b = m.group(2)
            sz = re.search(r'^\s*SIZE\s+(\S+)\s+BY\s+(\S+)', b, re.M)
            og = re.search(r'^\s*ORIGIN\s+(\S+)\s+(\S+)', b, re.M)
            pins = {pm.group(1): self._shapes(pm.group(2)) for pm in
                    re.finditer(r'^[ \t]*PIN[ \t]+(\S+)[ \t]*\r?\n(.*?)^[ \t]*END[ \t]+\1\b', b, re.M | re.S)}
            ob = re.search(r'^[ \t]*OBS\b(.*?)^[ \t]*END\b', b, re.M | re.S)
            self.macros[m.group(1)] = dict(size=(_num(sz.group(1)), _num(sz.group(2))) if sz else (0., 0.),
                                           origin=(_num(og.group(1)), _num(og.group(2))) if og else (0., 0.),
                                           pins=pins, obs=self._shapes(ob.group(1)) if ob else [])

    @staticmethod
    def _shapes(body):
        out, layer = [], None
        for line in body.splitlines():
            s = line.strip()
            m = re.match(r'^LAYER\s+(\S+)', s)
            if m:
                layer = m.group(1); continue
            m = re.match(r'^(RECT|POLYGON)\b(?:\s+MASK\s+\d+)?((?:\s+-?[\d.]+)+)', s)
            if m and layer:
                v = [float(x) for x in m.group(2).split()]
                if len(v) >= 4:
                    rs = poly_rects(v) if m.group(1) == 'POLYGON' else [(min(v[0::2]), min(v[1::2]), max(v[0::2]), max(v[1::2]))]
                    out += [(layer, r) for r in rs]
        return out

    def via_for(self, lo, hi):
        """a fixed via joining routing layers lo and hi (adjacent in the stack), preferring the flow's generated ones"""
        best = None
        for name, sh in self.vias.items():
            ls = {l for l, _ in sh}
            if lo in ls and hi in ls and not (ls - {lo, hi} - {l for l in ls if self.types.get(l) == 'CUT'}):
                if best is None or name.endswith('_FLOW'):
                    best = name
        return best


# ---------------------------------------------------------------------------------------------------- DEF
class Design(object):
    def __init__(self, path, tech):
        self.path, self.tech = path, tech
        self.text = Path(path).read_text(errors='replace')
        t = self.text
        m = re.search(r'^\s*UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', t, re.M); self.dbu = int(m.group(1)) if m else 1000
        m = re.search(r'DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)', t)
        self.die = tuple(int(v) for v in m.groups()) if m else (0, 0, 100000, 100000)
        g = re.findall(r'GCELLGRID\s+([XY])\s+(-?\d+)\s+DO\s+(\d+)\s+STEP\s+(\d+)', t)
        self.gcell = {a: (int(b), int(c), int(d)) for a, b, c, d in g}
        self.vias = {}   # DEF vias: name -> [(layer, rect dbu)]
        for st in self._stmts('VIAS'):
            self.vias[st.split()[1]] = self._def_via(st)
        self.comps = {}
        for st in self._stmts('COMPONENTS'):
            tk = TOK.findall(st)
            loc = None
            for j, x in enumerate(tk):
                if x in ('PLACED', 'FIXED', 'COVER') and j + 5 < len(tk) and tk[j + 1] == '(':
                    loc = (int(float(tk[j + 2])), int(float(tk[j + 3])), tk[j + 5]); break
            self.comps[tk[1]] = (tk[2], loc)
        self.iopins = {}   # name -> [(layer, rect dbu)]
        for st in self._stmts('PINS'):
            tk = TOK.findall(st); name = tk[1]; shapes, loc = [], None
            for j, x in enumerate(tk):
                if x == 'LAYER' and j + 1 < len(tk):
                    nums = [v for v in tk[j + 2:j + 16] if re.match(r'^-?\d+$', v)]
                    if len(nums) >= 4:
                        shapes.append((tk[j + 1], tuple(int(v) for v in nums[:4])))
                if x in ('PLACED', 'FIXED', 'COVER') and j + 5 < len(tk) and tk[j + 1] == '(' and loc is None:
                    loc = (int(tk[j + 2]), int(tk[j + 3]), tk[j + 5])
            if loc:
                out = []
                for layer, (a, b, c, d) in shapes:
                    p1, p2 = orient_off(a, b, loc[2]), orient_off(c, d, loc[2])
                    out.append((layer, (loc[0] + min(p1[0], p2[0]), loc[1] + min(p1[1], p2[1]),
                                        loc[0] + max(p1[0], p2[0]), loc[1] + max(p1[1], p2[1]))))
                self.iopins[name] = out
        self.nets = {}      # name -> dict(conns, shapes [(layer, rect)], span (start, end) in text, use)
        self.special = []   # [(layer, rect)]
        self._nets()
        self.blockages = []   # DEF BLOCKAGES on layers: [(layer, rect dbu, component or '', id blockage#k)]
        for st in self._stmts('BLOCKAGES'):
            tk = TOK.findall(st)
            if len(tk) < 3 or tk[1] != 'LAYER':
                continue   # placement blockages do not block wires
            lay = tk[2]; comp = tk[tk.index('COMPONENT') + 1] if 'COMPONENT' in tk else ''
            i = 3
            while i < len(tk):
                if tk[i] in ('RECT', 'POLYGON'):
                    j = i + 1; v = []
                    while j < len(tk) and tk[j] not in ('RECT', 'POLYGON', '+', ';'):
                        if re.match(r'^-?\d+$', tk[j]):
                            v.append(int(tk[j]))
                        j += 1
                    if len(v) >= 4:
                        rs = poly_rects(v) if tk[i] == 'POLYGON' else [(min(v[0::2]), min(v[1::2]), max(v[0::2]), max(v[1::2]))]
                        k = len({x[3] for x in self.blockages})
                        self.blockages += [(lay, r, comp, f'blockage#{k}') for r in rs]
                    i = j; continue
                i += 1
        self.drop = []   # obstacle drop specs (see obstacle_items / drop_match)

    def _section_span(self, name):
        m = re.search(r'^[ \t]*' + name + r'[ \t]+\d+[ \t]*;[^\n]*\n', self.text, re.M)
        if not m:
            return None
        e = re.search(r'^[ \t]*END[ \t]+' + name + r'\b', self.text[m.end():], re.M)
        return (m.end(), m.end() + e.start()) if e else None

    def _stmts(self, name):
        sp = self._section_span(name)
        if not sp:
            return []
        return [s.strip() for s in re.split(r'(?<=;)\s*\n', self.text[sp[0]:sp[1]]) if s.strip().startswith('-')]

    def _def_via(self, st):
        rects = [(m.group(1), tuple(int(float(v)) for v in re.findall(r'-?\d+', m.group(2))[:4]))
                 for m in re.finditer(r'\+\s*RECT\s+(\S+)((?:\s*\(\s*-?\d+\s+-?\d+\s*\)){2})', st)]
        if rects:
            return rects
        lay = re.search(r'\+\s*LAYERS\s+(\S+)\s+(\S+)\s+(\S+)', st)
        if not lay:
            return []
        n = lambda k, c: [int(v) for v in (re.search(r'\+\s*' + k + r'((?:\s+-?\d+){%d})' % c, st) or [None, ' 0' * c])[1].split()]
        cw, ch = n('CUTSIZE', 2); sx, sy = n('CUTSPACING', 2); bex, bey, tex, tey = n('ENCLOSURE', 4)
        rows, cols = n('ROWCOL', 2) if re.search(r'ROWCOL', st) else (1, 1)
        w = cols * cw + (cols - 1) * sx; h = rows * ch + (rows - 1) * sy
        bot, cut, top = lay.groups()
        return [(bot, (-w // 2 - bex, -h // 2 - bey, w - w // 2 + bex, h - h // 2 + bey)),
                (cut, (-w // 2, -h // 2, w - w // 2, h - h // 2)),
                (top, (-w // 2 - tex, -h // 2 - tey, w - w // 2 + tex, h - h // 2 + tey))]

    def via_shapes(self, name):
        if name in self.vias:
            return self.vias[name]
        u = self.dbu
        return [(l, tuple(int(round(v * u)) for v in r)) for l, r in self.tech.vias.get(name, [])]

    def _width(self, layer):
        L = self.tech.routing.get(layer)
        return int(round(L['width'] * self.dbu)) if L else 0

    def _wiring(self, tk, i, special):
        """wire rects and via shapes of a wiring token list starting at index i"""
        shapes = []; layer = None; prev = None; routing = False; width = 0
        while i < len(tk):
            x = tk[i]
            if x == '+':
                kw = tk[i + 1] if i + 1 < len(tk) else ''
                if kw in ('ROUTED', 'FIXED', 'COVER', 'NOSHIELD', 'SHIELD'):
                    routing = True; i += 2
                    if kw == 'SHIELD':
                        i += 1
                    layer = tk[i] if i < len(tk) else None; width = self._width(layer); prev = None; i += 1
                    if special and i < len(tk) and re.match(r'^\d+$', tk[i]):
                        width = int(tk[i]); i += 1
                    continue
                if routing and kw in ('SHAPE', 'MASK'):
                    i += 3; continue
                if kw in ('RECT', 'POLYGON') and i + 2 < len(tk):   # special-net shapes: + RECT layer ( a b ) ( c d )
                    j = i + 3; v = []
                    while j < len(tk) and tk[j] != '+' and tk[j] != ';':
                        if re.match(r'^-?\d+$', tk[j]):
                            v.append(int(tk[j]))
                        j += 1
                    if len(v) >= 4:
                        shapes.append((tk[i + 2], (min(v[0::2]), min(v[1::2]), max(v[0::2]), max(v[1::2]))))
                    routing = False; i = j; continue
                routing = False; i += 2; continue
            if not routing:
                i += 1; continue
            if x == 'NEW':
                layer = tk[i + 1]; width = self._width(layer); prev = None; i += 2
                if special and i < len(tk) and re.match(r'^\d+$', tk[i]):
                    width = int(tk[i]); i += 1
                continue
            if x == '(':
                j = tk.index(')', i); v = tk[i + 1:j]
                px = prev[0] if v[0] == '*' and prev else int(float(v[0]))
                py = prev[1] if len(v) > 1 and v[1] == '*' and prev else int(float(v[1]))
                cur = (px, py)
                if prev and prev != cur and width:
                    hw = width // 2
                    shapes.append((layer, (min(prev[0], px) - hw, min(prev[1], py) - hw, max(prev[0], px) + hw, max(prev[1], py) + hw)))
                prev = cur; i = j + 1; continue
            if x == 'RECT':
                j = tk.index(')', i); j2 = tk.index(')', j + 1); v = [int(float(a)) for a in tk[i + 1:j2] if re.match(r'^-?\d', a)]
                if prev and len(v) >= 4:
                    shapes.append((layer, (prev[0] + min(v[0], v[2]), prev[1] + min(v[1], v[3]), prev[0] + max(v[0], v[2]), prev[1] + max(v[1], v[3]))))
                i = j2 + 1; continue
            if x == 'VIRTUAL':
                j = tk.index(')', i); v = tk[i + 2:j]; prev = (int(float(v[0])), int(float(v[1]))); i = j + 1; continue
            if x in ('MASK', 'STYLE', 'TAPERRULE'):
                i += 2; continue
            if x == 'TAPER' or x in ORIENTS or re.match(r'^-?\d+$', x):
                i += 1; continue
            if prev:   # a via at the current point
                vs = self.via_shapes(x)
                for l, (a, b, c, d) in vs:
                    shapes.append((l, (prev[0] + a, prev[1] + b, prev[0] + c, prev[1] + d)))
                other = [l for l, _ in vs if l in self.tech.routing and l != layer]
                if other:
                    layer = other[0]; width = self._width(layer)
            i += 1
        return shapes

    def _nets(self):
        sp = self._section_span('NETS')
        if sp:
            for m in re.finditer(r'^[ \t]*-[ \t]+(\S+)(.*?);[ \t]*$', self.text[sp[0]:sp[1]], re.M | re.S):
                st = m.group(0); tk = TOK.findall(st); i = 2; conns = []
                while i < len(tk) and tk[i] == '(':
                    j = tk.index(')', i)
                    if j - i >= 3:
                        conns.append((tk[i + 1], tk[i + 2]))
                    i = j + 1
                use = re.search(r'\+\s*USE\s+(\S+)', st)
                self.nets[m.group(1)] = dict(conns=conns, shapes=self._wiring(tk, i, False),
                                             span=(sp[0] + m.start(), sp[0] + m.end()), use=use.group(1) if use else '')
        for st in self._stmts('SPECIALNETS'):
            self.special += self._wiring(TOK.findall(st), 2, True)

    def net_pieces(self, name):
        """the net's existing wiring as pieces: dict(kind seg|via|rect, layer, pts, via, rel, shapes, length)"""
        s, e = self.nets[name]['span']; tk = TOK.findall(self.text[s:e]); i = 2
        while i < len(tk) and tk[i] == '(':
            i = tk.index(')', i) + 1
        out = []; layer = None; prev = None; routing = False; width = 0
        while i < len(tk):
            x = tk[i]
            if x == '+':
                kw = tk[i + 1] if i + 1 < len(tk) else ''
                if kw in ('ROUTED', 'FIXED', 'COVER', 'NOSHIELD', 'SHIELD'):
                    routing = True; i += 3 if kw == 'SHIELD' else 2
                    layer = tk[i] if i < len(tk) else None; width = self._width(layer); prev = None; i += 1
                    continue
                if routing and kw in ('SHAPE', 'MASK'):
                    i += 3; continue
                routing = False; i += 2; continue
            if not routing:
                i += 1; continue
            if x == 'NEW':
                layer = tk[i + 1]; width = self._width(layer); prev = None; i += 2; continue
            if x == '(':
                j = tk.index(')', i); v = tk[i + 1:j]
                px = prev[0] if v[0] == '*' and prev else int(float(v[0]))
                py = prev[1] if len(v) > 1 and v[1] == '*' and prev else int(float(v[1]))
                cur = (px, py)
                if prev and prev != cur and width:
                    hw = width // 2
                    out.append(dict(kind='seg', layer=layer, pts=[prev, cur], length=abs(px - prev[0]) + abs(py - prev[1]),
                                    shapes=[(layer, (min(prev[0], px) - hw, min(prev[1], py) - hw, max(prev[0], px) + hw, max(prev[1], py) + hw))]))
                prev = cur; i = j + 1; continue
            if x == 'RECT':
                j = tk.index(')', i); j2 = tk.index(')', j + 1); v = [int(float(a)) for a in tk[i + 1:j2] if re.match(r'^-?\d', a)]
                if prev and len(v) >= 4:
                    rel = (min(v[0], v[2]), min(v[1], v[3]), max(v[0], v[2]), max(v[1], v[3]))
                    out.append(dict(kind='rect', layer=layer, pts=[prev], rel=rel, length=0,
                                    shapes=[(layer, (prev[0] + rel[0], prev[1] + rel[1], prev[0] + rel[2], prev[1] + rel[3]))]))
                i = j2 + 1; continue
            if x == 'VIRTUAL':
                j = tk.index(')', i); v = tk[i + 2:j]; prev = (int(float(v[0])), int(float(v[1]))); i = j + 1; continue
            if x in ('MASK', 'STYLE', 'TAPERRULE'):
                i += 2; continue
            if x == 'TAPER' or x in ORIENTS or re.match(r'^-?\d+$', x):
                i += 1; continue
            if prev:
                vs = self.via_shapes(x)
                out.append(dict(kind='via', layer=layer, pts=[prev], via=x, length=0,
                                shapes=[(l, (prev[0] + a, prev[1] + b, prev[0] + c, prev[1] + d)) for l, (a, b, c, d) in vs]))
                other = [l for l, _ in vs if l in self.tech.routing and l != layer]
                if other:
                    layer = other[0]; width = self._width(layer)
            i += 1
        return out

    def pin_shapes(self, inst, pin):
        """[(layer, rect dbu)] of an instance pin (or an I/O pin with inst 'PIN')"""
        if inst == 'PIN':
            return self.iopins.get(pin, [])
        mac, loc = self.comps.get(inst, (None, None))
        M = self.tech.macros.get(mac)
        if not M or not loc:
            return []
        return self._place(M, M['pins'].get(pin) or M['pins'].get(pin.replace('\\', '')) or [], loc)

    def _place(self, M, shapes, loc):
        w, h = M['size']; ox, oy = M['origin']; u = self.dbu; out = []
        for layer, (a, b, c, d) in shapes:
            p1 = orient_pt(a + ox, b + oy, w, h, loc[2]); p2 = orient_pt(c + ox, d + oy, w, h, loc[2])
            out.append((layer, (int(round(loc[0] + min(p1[0], p2[0]) * u)), int(round(loc[1] + min(p1[1], p2[1]) * u)),
                                int(round(loc[0] + max(p1[0], p2[0]) * u)), int(round(loc[1] + max(p1[1], p2[1]) * u)))))
        return out

    def obstacle_items(self, skip_net):
        """every shape the reroute of skip_net must keep spacing to: [(layer, rect, owner label, kind, inst, macro)];
        kind: special | net | obs (LEF cell OBS) | pin (another net's cell pin) | iopin | blockage (DEF BLOCKAGES)"""
        out = [(l, r, 'special net (power/ground)', 'special', '', '') for l, r in self.special]
        own = set(self.nets.get(skip_net, {}).get('conns', []))
        for n, v in self.nets.items():
            if n != skip_net:
                out += [(l, r, f'net {n}', 'net', '', '') for l, r in v['shapes']]
        for inst, (mac, loc) in self.comps.items():
            M = self.tech.macros.get(mac)
            if not M or not loc:
                continue
            out += [(l, r, f'{inst} OBS', 'obs', inst, mac) for l, r in self._place(M, M['obs'], loc)]
            for pin, shapes in M['pins'].items():
                if (inst, pin) not in own:
                    out += [(l, r, f'pin {inst}/{pin}', 'pin', inst, mac) for l, r in self._place(M, shapes, loc)]
        for pin, shapes in self.iopins.items():
            if ('PIN', pin) not in own:
                out += [(l, r, f'I/O pin {pin}', 'iopin', '', '') for l, r in shapes]
        out += [(l, r, f'DEF {k}' + (f' of {c}' if c else ''), 'blockage', k, '') for l, r, c, k in self.blockages]
        return out

    def drop_match(self, it, skip_net, specs=None):
        """is this obstacle item dropped by the specs (default self.drop)? A spec is one of
        all (every cell OBS) | LAYER (cell OBS on that layer) | own[:LAYER] (OBS of the cells the net connects to) |
        inst=NAME[:LAYER] | MACRO[:LAYER] (cell OBS of a LEF macro; * and ? allowed) | blockage[:LAYER] (DEF BLOCKAGES) |
        blockage#K (the K-th DEF blockage, numbered from 0 as in the diagnosis)"""
        from fnmatch import fnmatchcase
        fm = lambda v, pat: v == pat or fnmatchcase(v, pat)   # names with [ ] (buses) match exactly
        l, _, _, kind, inst, mac = it
        for s in (self.drop if specs is None else specs):
            a, _, b = s.partition(':')
            if b and not fm(l, b):
                continue
            if a == 'blockage' or a.startswith('blockage#'):   # every DEF blockage, or one by its number
                if kind == 'blockage' and (a == 'blockage' or inst == a):
                    return True
                continue
            if kind != 'obs':
                continue
            if a == 'all' or (a in self.tech.types and not b and l == a):
                return True
            if a == 'own' and inst in {i for i, _ in self.nets.get(skip_net, {}).get('conns', [])}:
                return True
            if a.startswith('inst=') and fm(inst, a[5:]):
                return True
            if a not in ('all', 'own') and not a.startswith('inst=') and a not in self.tech.types and fm(mac, a):
                return True
        return False

    def obstacles_labeled(self, skip_net):
        """[(layer, rect, owner)] of every obstacle (dropped ones included: for the diagnosis)"""
        return [(l, r, o) for l, r, o, _, _, _ in self.obstacle_items(skip_net)]

    def obstacles(self, skip_net):
        """[(layer, rect)] every shape the reroute of skip_net must keep spacing to (without the dropped OBS)"""
        return [(it[0], it[1]) for it in self.obstacle_items(skip_net) if not (self.drop and self.drop_match(it, skip_net))]


# ---------------------------------------------------------------------------------------------------- routing
class Router(object):
    def __init__(self, tech, design, layers):
        self.T, self.D = tech, design
        self.L = layers                                   # signal layers, bottom first
        u = design.dbu
        # each layer routes on its own tracks (pitch from the LEF; offset from where the design's own wires on that layer
        # run, else the LEF OFFSET, else half a pitch); nodes sit where tracks of the layers cross, so another net's wire
        # blocks only the tracks it really occupies
        self.P = min(int(round(tech.routing[l]['pitch'] * u)) for l in layers) or 100
        self.pitch = {l: max(1, int(round(tech.routing[l]['pitch'] * u)), int(np.ceil((tech.routing[l]['width'] + tech.routing[l]['spacing']) * u - 1e-9)))
                      for l in layers}
        self.toff = {l: self._offset(l) for l in layers}
        self.below = None                                  # routing layer just below the bottom signal layer
        i = tech.rorder.index(layers[0])
        if i > 0:
            self.below = tech.rorder[i - 1]
        self.hor = [tech.routing[l]['dir'].startswith('H') for l in layers]

    def _offset(self, l):
        T, D = self.T, self.D; u = D.dbu; R = T.routing[l]; p = self.pitch[l]; w = int(round(R['width'] * u))
        hor = R['dir'].startswith('H'); base = D.die[1] if hor else D.die[0]
        cnt = {}
        for v in D.nets.values():
            for ll, r in v['shapes']:
                if ll != l:
                    continue
                a, b = r[2] - r[0], r[3] - r[1]
                if hor and a > b and b <= 2 * w:
                    c, n = (r[1] + r[3]) // 2, a
                elif not hor and b > a and a <= 2 * w:
                    c, n = (r[0] + r[2]) // 2, b
                else:
                    continue
                k = (c - base) % p; cnt[k] = cnt.get(k, 0) + n
        if cnt:
            return max(cnt, key=cnt.get)
        if R.get('offset') is not None:
            return int(round(R['offset'] * u)) % p
        return p // 2

    def _tracks(self, l, lo, hi):
        p = self.pitch[l]; base = (self.D.die[1] if self.T.routing[l]['dir'].startswith('H') else self.D.die[0]) + self.toff[l]
        n0 = -(-(lo - base) // p)
        return list(range(base + n0 * p, hi + 1, p))

    def _lattice(self, win):
        x1, y1, x2, y2 = win
        xs, ys = set(), set(); self.on = []
        for k, l in enumerate(self.L):
            t = self._tracks(l, y1, y2) if self.hor[k] else self._tracks(l, x1, x2)
            (ys if self.hor[k] else xs).update(t); self.on.append(set(t))
        if not xs:
            xs = set(range(x1, x2 + 1, self.P))
        if not ys:
            ys = set(range(y1, y2 + 1, self.P))
        self.X, self.Y = sorted(xs), sorted(ys)
        return len(self.X), len(self.Y)

    def xy(self, i, j):
        return self.X[i], self.Y[j]

    def _rng(self, axis, lo, hi):   # indices of lattice coordinates within [lo, hi]
        import bisect
        A = self.X if axis == 0 else self.Y
        return bisect.bisect_left(A, lo), bisect.bisect_right(A, hi) - 1

    def _near(self, axis, v):
        import bisect
        A = self.X if axis == 0 else self.Y
        k = bisect.bisect_left(A, v)
        return min((c for c in (k - 1, k) if 0 <= c < len(A)), key=lambda c: abs(A[c] - v))

    def _mark(self, arr, rect, grow):
        x1, y1, x2, y2 = rect
        a, b = self._rng(0, x1 - grow, x2 + grow); c, d = self._rng(1, y1 - grow, y2 + grow)
        a, c = max(a, 0), max(c, 0); b, d = min(b, arr.shape[0] - 1), min(d, arr.shape[1] - 1)
        if a <= b and c <= d:
            arr[a:b + 1, c:d + 1] = True

    def route_net(self, net, guide=None, margin=12, terms=None):
        """terms: [((inst, pin), shapes)] to connect instead of the net's pins (e.g. pins with their kept access wiring)"""
        D, T, u = self.D, self.T, self.D.dbu
        info = D.nets[net]
        self.capped = False
        self.max_expand = int(__import__('os').environ.get('ROUTER_MAX_EXPAND', '4000000'))
        pins = []
        for inst, pin in ([] if terms else info['conns']):
            sh = [(l, r) for l, r in D.pin_shapes(inst, pin) if l in T.routing]
            if sh:
                pins.append(((inst, pin), sh))
        if terms:
            pins = [(lab, [(l, r) for l, r in sh if l in T.routing]) for lab, sh in terms]
        if len(pins) < 2:
            raise NoPath(f'{net}: fewer than 2 pins with shapes on routing layers', None)
        pts = [r for _, sh in pins for _, r in sh] + ([g[:4] for g in guide] if guide else [])
        win = (min(r[0] for r in pts) - margin * self.P, min(r[1] for r in pts) - margin * self.P,
               max(r[2] for r in pts) + margin * self.P, max(r[3] for r in pts) + margin * self.P)
        nx, ny = self._lattice(win)
        L = len(self.L)
        blocked = np.zeros((L, nx, ny), bool); viablk = np.zeros((L, nx, ny), bool)
        obs = D.obstacles(net)
        vpad = {}
        for k, l in enumerate(self.L):
            R = T.routing[l]; hw = int(round(R['width'] * u / 2)); sp = int(round(R['spacing'] * u))
            pad = hw
            for lo, hi in zip(self.L[:-1], self.L[1:]):
                if l in (lo, hi):
                    vn = T.via_for(lo, hi)
                    for ll, r in T.vias.get(vn, []):
                        if ll == l:
                            pad = max(pad, int(round(max(r[2] - r[0], r[3] - r[1]) * u / 2)))
            vpad[l] = pad
            if self.hor[k]:   # lattice points that are not on this layer's own tracks
                blocked[k][:, [j for j in range(ny) if self.Y[j] not in self.on[k]]] = True
            else:
                blocked[k][[i for i in range(nx) if self.X[i] not in self.on[k]], :] = True
            for ol, r in obs:
                if ol == l:
                    self._mark(blocked[k], r, sp + hw - 1)
                    self._mark(viablk[k], r, sp + pad - 1)
        # pin layer below the signal range: where the stack via's pad there keeps spacing to other shapes
        belowblk = np.zeros((nx, ny), bool)
        if self.below:
            vn = T.via_for(self.below, self.L[0]); Rb = T.routing[self.below]
            padb = max([int(round(max(r[2] - r[0], r[3] - r[1]) * u / 2)) for ll, r in T.vias.get(vn, []) if ll == self.below]
                       or [int(round(Rb['width'] * u / 2))])
            spb = int(round(Rb['spacing'] * u))
            for ol, r in obs:
                if ol == self.below:
                    self._mark(belowblk, r, spb + padb - 1)
        # exact spacing checks for off-grid pin access
        obs_by = {}
        for ol, r in obs:
            obs_by.setdefault(ol, []).append(r)

        def clear(layer, rect):
            R = T.routing.get(layer)
            if not R:
                return True
            sp = int(round(R['spacing'] * u))
            return not any(rect[0] < o[2] + sp and o[0] < rect[2] + sp and rect[1] < o[3] + sp and o[1] < rect[3] + sp
                           for o in obs_by.get(layer, ()))
        mg = max(1, int(round(T.mgrid * u)))
        snap = lambda v: int(round(v / mg)) * mg
        vstack = T.via_for(self.below, self.L[0]) if self.below else None
        vsh = [(ll, tuple(int(round(c * u)) for c in r)) for ll, r in T.vias.get(vstack, [])]
        w0 = int(round(T.routing[self.L[0]]['width'] * u / 2))

        def offgrid(rects):
            """access points on the pin itself: (node, point, stub corner) with exact via-pad and stub spacing"""
            out = []
            for r in rects:
                cx, cy = (r[0] + r[2]) / 2., (r[1] + r[3]) / 2.
                horiz = (r[2] - r[0]) >= (r[3] - r[1])
                half = ((r[2] - r[0]) if horiz else (r[3] - r[1])) / 2.
                for t in (0., -.25, .25, -.45, .45):
                    p = (snap(cx + t * 2 * half), snap(cy)) if horiz else (snap(cx), snap(cy + t * 2 * half))
                    if not all(clear(ll, (p[0] + a, p[1] + b, p[0] + c, p[1] + d)) for ll, (a, b, c, d) in vsh
                               if ll in (self.below, self.L[0])):
                        continue
                    i0 = self._near(0, p[0]); j0 = self._near(1, p[1])
                    got = 0
                    for di, dj in sorted(((a, b) for a in range(-2, 3) for b in range(-2, 3)), key=lambda v: abs(v[0]) + abs(v[1])):
                        i, j = i0 + di, j0 + dj
                        if not (0 <= i < nx and 0 <= j < ny) or blocked[0, i, j]:
                            continue
                        x, y = self.xy(i, j)
                        for corner in ((x, p[1]), (p[0], y)):   # L-shaped stub, either way round
                            stub = [(min(p[0], corner[0]) - w0, min(p[1], corner[1]) - w0, max(p[0], corner[0]) + w0, max(p[1], corner[1]) + w0),
                                    (min(corner[0], x) - w0, min(corner[1], y) - w0, max(corner[0], x) + w0, max(corner[1], y) + w0)]
                            if all(clear(self.L[0], s) for s in stub):
                                out.append((0, i, j, ('S', p[0], p[1], self.below, corner[0], corner[1])))
                                got += 1
                                break
                        if got >= 2:
                            break
                    if len(out) >= 6:
                        return out
            return out
        # pin access: nodes on the bottom signal layer over the pin (via stack down to it) or on the pin's layer
        targets = []
        for (inst, pin), sh in pins:
            acc = set()
            for l, r in sh:
                k = self.L.index(l) if l in self.L else (0 if l == self.below else None)
                if k is None:
                    continue
                a, b = self._rng(0, r[0], r[2]); c, d = self._rng(1, r[1], r[3])
                if l in self.L and (a > b or c > d):   # thinner than a pitch, off the grid: nodes next to it + a short jog
                    hw = int(round(T.routing[l]['width'] * u / 2))
                    for i in range(max(min(a, b) - 1, 0), min(max(a, b) + 1, nx - 1) + 1):
                        for j in range(max(min(c, d) - 1, 0), min(max(c, d) + 1, ny - 1) + 1):
                            x, y = self.xy(i, j)
                            mx_, my_ = min(hw, (r[2] - r[0]) // 2), min(hw, (r[3] - r[1]) // 2)
                            px = min(max(x, r[0] + mx_), r[2] - mx_); py = min(max(y, r[1] + my_), r[3] - my_)
                            if abs(px - x) + abs(py - y) > 2 * self.P:
                                continue
                            legs = [(min(x, px) - hw, y - hw, max(x, px) + hw, y + hw), (px - hw, min(y, py) - hw, px + hw, max(y, py) + hw)]
                            if all(clear(l, g) for g in legs):
                                acc.add((k, i, j, ('J', k, px, py)))
                    continue
                if a > b or c > d:   # pad smaller than a pitch: the node nearest its centre
                    a = b = self._near(0, (r[0] + r[2]) // 2)
                    c = d = self._near(1, (r[1] + r[3]) // 2)
                for i in range(max(a, 0), min(b, nx - 1) + 1):
                    for j in range(max(c, 0), min(d, ny - 1) + 1):
                        acc.add((k, i, j, l if l not in self.L else None))
            low = [r for l, r in sh if l == self.below]
            if low and vsh and not any(l in self.L for l, _ in sh):
                og = offgrid(low)
                if og:   # off-grid access on the pin with exact spacing: preferred over grid nodes
                    acc = set(og)
            if not acc:
                raise NoPath(f'{net}: no access point for {inst}/{pin}', (inst, pin))
            # the net's own shapes are no obstacles, so a blocked access node is blocked by another net: use free ones
            # (on the pin layer below, also where the stack via's pad keeps spacing); only if there is none, keep the
            # best ones and let the self-check report it
            for test in (lambda a: not blocked[a[0], a[1], a[2]] and (a[3] is None or isinstance(a[3], tuple)
                                                                         or (not belowblk[a[1], a[2]] and not viablk[a[0], a[1], a[2]])),
                         lambda a: not blocked[a[0], a[1], a[2]],
                         lambda a: True):
                clean = {a for a in acc if test(a)}
                if clean:
                    acc = clean
                    break
            for k, i, j, _ in acc:
                blocked[k, i, j] = False
            targets.append(acc)
        gmask = None
        if guide:
            gmask = np.zeros((nx, ny), bool)
            for g in guide:
                self._mark(gmask, g[:4], 0)
        # grow a tree from the first pin: its first path starts at one of the pin's access nodes, and from then on
        # the tree holds only nodes that are really wired (a via stack goes down at every pin node a path ends on)
        first = {(k, i, j): below for k, i, j, below in targets[0]}
        tree = set(first)
        stacks, path_nodes = [], []
        rest = list(range(1, len(targets)))
        cen = lambda t: (sum(i for _, i, _, _ in targets[t]) / len(targets[t]), sum(j for _, _, j, _ in targets[t]) / len(targets[t]))
        while rest:
            tl = list(tree)[::max(1, len(tree) // 300)]
            ti = min(rest, key=lambda t: min(abs(cen(t)[0] - a) + abs(cen(t)[1] - b) for _, a, b in tl))
            rest.remove(ti)
            goal = {(k, i, j): below for k, i, j, below in targets[ti]}
            p = self._astar(tree, goal, blocked, viablk, gmask, nx, ny)
            if p is None:   # no legal landing: allow the via pad at the pin (the self-check reports it)
                p = self._astar(tree, goal, blocked, viablk, gmask, nx, ny, relax_goal_via=True)
            if p is None:
                def why(nodes):
                    free = sum(1 for k, i, j in nodes for dk, di, dj in ((0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0))
                               if 0 <= k + dk < L and 0 <= i + di < nx and 0 <= j + dj < ny and not blocked[k + dk, i + di, j + dj])
                    lay = sorted({self.L[k] for k, _, _ in nodes})
                    return f'{len(nodes)} access node(s) on {"/".join(lay)}, {free} free neighbour(s)'
                raise NoPath(f'{net}: no path to pin {pins[ti][0][0]}/{pins[ti][0][1]} ({why(list(goal))}; '
                             f'from {pins[0][0][0]}/{pins[0][0][1]}: {why(list(first))}; grid {nx}x{ny}x{L}, pitch {self.P})'
                             + (f' [SEARCH LIMIT: stopped after {self.max_expand} expanded nodes without reaching the pin, so a path '
                                f'may still exist; ROUTER_MAX_EXPAND raises the limit]' if self.capped else ''), pins[ti][0], pins[0][0])
            if not path_nodes:
                tree = set()
                if first.get(p[0]):
                    stacks.append((p[0][1], p[0][2], first[p[0]]))
            if goal.get(p[-1]):
                stacks.append((p[-1][1], p[-1][2], goal[p[-1]]))
            path_nodes.append(p)
            tree |= set(p)
        return self._wiring(path_nodes, stacks)

    def _astar(self, srcs, goal, blocked, viablk, gmask, nx, ny, relax_goal_via=False):
        L = len(self.L); X, Y, P = self.X, self.Y, float(self.P)
        gx = [X[i] for _, i, _ in goal]; gy = [Y[j] for _, _, j in goal]
        bx0, bx1, by0, by1 = min(gx), max(gx), min(gy), max(gy)
        h = lambda i, j: (max(bx0 - X[i], 0, X[i] - bx1) + max(by0 - Y[j], 0, Y[j] - by1)) / P
        dist = {}; par = {}; pq = []
        for s in srcs:
            dist[s] = 0.; par[s] = None; heapq.heappush(pq, (h(s[1], s[2]), 0., s))
        n = 0
        while pq:
            f, g, cur = heapq.heappop(pq)
            if g > dist.get(cur, 1e18):
                continue
            if cur in goal:
                out = [cur]
                while par[out[-1]] is not None:
                    out.append(par[out[-1]])
                return out[::-1]
            n += 1
            if n > self.max_expand:
                self.capped = True   # gave up: a path may still exist (reported, not taken for a blocked pin)
                return None
            k, i, j = cur
            hor = self.hor[k]
            for dk, di, dj in ((0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0)):
                kk, ii, jj = k + dk, i + di, j + dj
                if not (0 <= kk < L and 0 <= ii < nx and 0 <= jj < ny):
                    continue
                if dk:
                    if (viablk[k, i, j] or viablk[kk, ii, jj]) and not (relax_goal_via and (kk, ii, jj) in goal):
                        continue
                    if blocked[kk, ii, jj] and (kk, ii, jj) not in goal:
                        continue
                    c = 4.
                else:
                    if blocked[kk, ii, jj] and (kk, ii, jj) not in goal:
                        continue
                    c = (abs(X[ii] - X[i]) + abs(Y[jj] - Y[j])) / P * (1. if (di != 0) == hor else 3.)
                if gmask is not None and not gmask[ii, jj]:
                    c += 3.
                ng = g + c
                nb = (kk, ii, jj)
                if ng < dist.get(nb, 1e18):
                    dist[nb] = ng; par[nb] = cur
                    heapq.heappush(pq, (ng + h(ii, jj), ng, nb))
        return None

    def _wiring(self, paths, stacks):
        """DEF wiring pieces and the new shapes (for the self-check) from node paths"""
        pieces, shapes = [], []
        u = self.D.dbu
        for p in paths:
            run = [p[0]]
            for a, b in zip(p, p[1:]):
                if a[0] != b[0]:   # via
                    self._flush(run, pieces, shapes)
                    lo, hi = sorted((a[0], b[0]))
                    x, y = self.xy(a[1], a[2])
                    vn = self.T.via_for(self.L[lo], self.L[hi])
                    if not vn:
                        raise RuntimeError(f'no fixed via between {self.L[lo]} and {self.L[hi]}')
                    pieces.append((self.L[lo], [(x, y)], vn))
                    shapes += [(l, (x + int(round(r[0] * u)), y + int(round(r[1] * u)), x + int(round(r[2] * u)), y + int(round(r[3] * u))))
                               for l, r in self.T.vias[vn]]
                    run = [b]
                else:
                    run.append(b)
            self._flush(run, pieces, shapes)
        for i, j, low in stacks:   # down from the bottom signal layer to a pin on the layer below
            x, y = self.xy(i, j)
            if isinstance(low, tuple) and low[0] == 'J':   # jog on the node's own layer to an off-grid shape of the terminal
                _, k, px, py = low; layer = self.L[k]; hw = int(round(self.T.routing[layer]['width'] * u / 2))
                pts = [(x, y)] + ([(px, y)] if (px, y) not in ((x, y), (px, py)) else []) + ([(px, py)] if (px, py) != (x, y) else [])
                if len(pts) > 1:
                    pieces.append((layer, pts, None))
                    for a, b in zip(pts, pts[1:]):
                        shapes.append((layer, (min(a[0], b[0]) - hw, min(a[1], b[1]) - hw, max(a[0], b[0]) + hw, max(a[1], b[1]) + hw)))
                continue
            if isinstance(low, tuple):   # off-grid access: via on the pin point, stub on the bottom layer to the node
                _, px, py, low, cx, cy = low
                pts = [(px, py)] + ([(cx, cy)] if (cx, cy) not in ((px, py), (x, y)) else []) + ([(x, y)] if (x, y) != (px, py) else [])
                if len(pts) > 1:
                    layer = self.L[0]; hw = int(round(self.T.routing[layer]['width'] * u / 2))
                    pieces.append((layer, pts, None))
                    for a, b in zip(pts, pts[1:]):
                        shapes.append((layer, (min(a[0], b[0]) - hw, min(a[1], b[1]) - hw, max(a[0], b[0]) + hw, max(a[1], b[1]) + hw)))
                x, y = px, py
            vn = self.T.via_for(low, self.L[0])
            if vn:
                pieces.append((low, [(x, y)], vn))
                shapes += [(l, (x + int(round(r[0] * u)), y + int(round(r[1] * u)), x + int(round(r[2] * u)), y + int(round(r[3] * u))))
                           for l, r in self.T.vias[vn]]
        return pieces, shapes

    def _flush(self, run, pieces, shapes):
        if len(run) < 2:
            return
        layer = self.L[run[0][0]]
        hw = int(round(self.T.routing[layer]['width'] * self.D.dbu / 2))
        pts = [self.xy(i, j) for _, i, j in run]
        corners = [pts[0]]
        for a, b, c in zip(pts, pts[1:], pts[2:]):
            if (a[0] == b[0]) != (b[0] == c[0]):
                corners.append(b)
        corners.append(pts[-1])
        pieces.append((layer, corners, None))
        for a, b in zip(corners, corners[1:]):
            shapes.append((layer, (min(a[0], b[0]) - hw, min(a[1], b[1]) - hw, max(a[0], b[0]) + hw, max(a[1], b[1]) + hw)))


def def_wiring(pieces):
    """'+ ROUTED ... NEW ...' text from (layer, points, via) pieces"""
    out = []
    for k, (layer, pts, via) in enumerate(pieces):
        head = '+ ROUTED ' if k == 0 else '\n      NEW '
        if isinstance(via, tuple):   # kept patch: ('RECT', (dx1, dy1, dx2, dy2))
            r = via[1]; out.append(f'{head}{layer} ( {pts[0][0]} {pts[0][1]} ) RECT ( {r[0]} {r[1]} {r[2]} {r[3]} )')
            continue
        if via:
            out.append(f'{head}{layer} ( {pts[0][0]} {pts[0][1]} ) {via}')
            continue
        s = f'{head}{layer} ( {pts[0][0]} {pts[0][1]} )'
        for a, b in zip(pts, pts[1:]):
            s += ' ( %s %s )' % ('*' if b[0] == a[0] else b[0], '*' if b[1] == a[1] else b[1])
        out.append(s)
    return ''.join(out)


def self_check(shapes, obstacles, tech, dbu, net):
    """spacing / short check of the new shapes against everything else -> DRT-style report lines"""
    out = []
    by = {}
    for l, r in obstacles:
        by.setdefault(l, []).append(r)
    for l, r in shapes:
        R = tech.routing.get(l)
        if not R:
            continue
        sp = int(round(R['spacing'] * dbu))
        for o in by.get(l, []):
            if r[0] < o[2] + sp and o[0] < r[2] + sp and r[1] < o[3] + sp and o[1] < r[3] + sp:
                kind = 'Short' if (r[0] < o[2] and o[0] < r[2] and r[1] < o[3] and o[1] < r[3]) else 'MetSpc'
                out.append(f'  violation type: {kind}\n    srcs: net:{net} obstacle:{o}\n    bbox = ( {r[0] / dbu} , {r[1] / dbu} ) - '
                           f'( {r[2] / dbu} , {r[3] / dbu} ) on Layer {l}')
                break
    return out


def _rdist(a, b):
    """gap between two rects (0 when they touch or overlap)"""
    return max(a[0] - b[2], b[0] - a[2], a[1] - b[3], b[1] - a[3], 0)


_AUDIT = {}
GROUP = {'obs': 'cell OBS (LEF)', 'blockage': 'DEF blockage', 'net': 'other nets', 'pin': 'other pins', 'iopin': 'other pins',
         'special': 'power/ground'}


def access_block(D, R, net, sh):
    """what keeps a via landing out over the pin: sample points on the pin shapes, and per layer (the pin's layers and
    the lowest signal layer above) the share of points where a via pad would be within spacing of each obstacle.
    Dropped OBS (D.drop) are left out. -> [dict(layer, n, blocked, groups {group: share}, owners [(share, owner, kind,
    inst)], under (share of points inside cell OBS, pin layers only))]"""
    T, u = D.tech, D.dbu
    items = [it for it in D.obstacle_items(net) if not (D.drop and D.drop_match(it, net))]
    pts = []
    step = max(1, R.P // 2)
    for _, r in sh:
        nx_ = max(1, min(9, (r[2] - r[0]) // step)); ny_ = max(1, min(9, (r[3] - r[1]) // step))
        pts += [(r[0] + (r[2] - r[0]) * (a + .5) / nx_, r[1] + (r[3] - r[1]) * (b + .5) / ny_) for a in range(nx_) for b in range(ny_)]
    vn = T.via_for(R.below, R.L[0]) if R.below else None
    out = []
    for lay in [l for l in dict.fromkeys([a[0] for a in sh] + [R.L[0]]) if l in T.routing]:
        Rl = T.routing[lay]; sp = Rl['spacing'] * u
        hp = max([Rl['width'] * u / 2] + [max(r[2] - r[0], r[3] - r[1]) * u / 2 for ll, r in T.vias.get(vn, []) if ll == lay])
        g = hp + sp
        bx = (min(p[0] for p in pts) - g, min(p[1] for p in pts) - g, max(p[0] for p in pts) + g, max(p[1] for p in pts) + g)
        near = [it for it in items if it[0] == lay and it[1][0] < bx[2] and bx[0] < it[1][2] and it[1][1] < bx[3] and bx[1] < it[1][3]]
        per, grp, nb, under = {}, {}, 0, 0
        for x, y in pts:
            hit = [it for it in near if it[1][0] < x + g and x - g < it[1][2] and it[1][1] < y + g and y - g < it[1][3]]
            nb += bool(hit)
            for k in {GROUP[it[3]] for it in hit}:
                grp[k] = grp.get(k, 0) + 1
            for it in {(it[2], it[3], it[4]) for it in hit}:
                per[it] = per.get(it, 0) + 1
            under += any(it[3] == 'obs' and it[1][0] < x < it[1][2] and it[1][1] < y < it[1][3] for it in near)
        n = len(pts) or 1
        out.append(dict(layer=lay, n=len(pts), blocked=nb / n, groups={k: v / n for k, v in grp.items()},
                        owners=sorted(((v / n, o, kd, i) for (o, kd, i), v in per.items()), reverse=True),
                        under=under / n if lay in {a[0] for a in sh} else 0.))
    return out


def obs_audit(D):
    """design-wide: how much of the cells the LEF OBS covers, how many connected pins lie under their own cell's OBS,
    and where the original wiring itself overlaps cell OBS / DEF blockages"""
    if 'o' in _AUDIT:
        return _AUDIT['o']
    T = D.tech; cov = {}; under = []; tot = 0
    for inst, (mac, loc) in D.comps.items():
        M = T.macros.get(mac)
        if not M or not loc or not M['obs']:
            continue
        w, h = M['size']; area = max(w * h, 1e-12)
        for l in {l for l, _ in M['obs'] if l in T.routing}:
            a = sum((r[2] - r[0]) * (r[3] - r[1]) for ll, r in M['obs'] if ll == l)   # overlaps counted twice: an upper bound
            cov.setdefault(l, {})[mac] = min(1., a / area)
    for n, v in D.nets.items():
        if v['use'] in ('POWER', 'GROUND'):
            continue
        for inst, pin in v['conns']:
            mac, loc = D.comps.get(inst, (None, None))
            M = T.macros.get(mac)
            if not M or not loc:
                continue
            ps = D.pin_shapes(inst, pin); tot += bool(ps)
            ob = D._place(M, M['obs'], loc)
            if any(a[0] == o[0] and a[1][0] < o[1][2] and o[1][0] < a[1][2] and a[1][1] < o[1][3] and o[1][1] < a[1][3] for a in ps for o in ob):
                under.append(f'{inst}/{pin} ({mac})')
    places, nets = 0, set()
    blk = {}
    for inst, (mac, loc) in D.comps.items():
        M = T.macros.get(mac)
        if M and loc and M['obs']:
            for l, r in D._place(M, M['obs'], loc):
                blk.setdefault(l, []).append(r)
    for l, r, _, _ in D.blockages:
        blk.setdefault(l, []).append(r)
    C = 4096; idx = {}   # bucket index of the OBS / blockage rects
    for l, rs in blk.items():
        for r in rs:
            for bx in range(r[0] // C, r[2] // C + 1):
                for by in range(r[1] // C, r[3] // C + 1):
                    idx.setdefault((l, bx, by), []).append(r)
    for n, v in D.nets.items():
        for l, r in v['shapes']:
            if l in T.routing and any(r[0] < o[2] and o[0] < r[2] and r[1] < o[3] and o[1] < r[3]
                                      for bx in range(r[0] // C, r[2] // C + 1) for by in range(r[1] // C, r[3] // C + 1)
                                      for o in idx.get((l, bx, by), ())):
                places += 1; nets.add(n)
    _AUDIT['o'] = dict(cov=cov, under=under, pins=tot, places=places, nets=sorted(nets))
    return _AUDIT['o']


def audit(D):
    """original layout: how many pin connections are reached by their own net's wiring (shapes touching on a layer)"""
    if 'r' in _AUDIT:
        return _AUDIT['r']
    tot = hit = 0; miss = []
    for n, v in D.nets.items():
        if not v['shapes'] or v['use'] in ('POWER', 'GROUND'):
            continue
        for inst, pin in v['conns']:
            sh = D.pin_shapes(inst, pin)
            if not sh:
                continue
            tot += 1
            if any(_touch(a, b) for a in sh for b in v['shapes']):
                hit += 1
            elif len(miss) < 4:
                miss.append(f'{n}:{inst}/{pin}')
    _AUDIT['r'] = (hit, tot, miss)
    return _AUDIT['r']


def diagnose(D, R, net, pin, info=None):
    """why a pin of the net cannot be connected: (verdict, reason, evidence lines). verdict: TESTCASE (the input layout),
    LEF/DEF (the LEF does not match the DEF, or the flow places cells wrongly), ROUTER (this router)"""
    T, u = D.tech, D.dbu
    um = lambda v: f'{v / u:.3f}'
    hit, tot, miss = audit(D)
    ev = [f'[design] original layout: {hit} of {tot} pin connections in the design are reached by their own net\'s wiring'
          + (f' (not reached e.g. {", ".join(miss)})' if miss else '')]
    oa = obs_audit(D)
    if oa['cov'] or D.blockages:
        lays = ', '.join(f'{l}: {len(m)} cell type(s), up to {100 * max(m.values()):.0f}% of the cell area'
                         for l, m in sorted(oa['cov'].items(), key=lambda kv: T.rorder.index(kv[0])))
        ev.append(f'[design] LEF OBS on routing layers: {lays or "none"}; {len(oa["under"])} of {oa["pins"]} connected cell pins overlap '
                  f'their own cell\'s OBS' + (f' (e.g. {", ".join(oa["under"][:3])})' if oa['under'] else '')
                  + f'; the original wiring overlaps cell OBS / DEF blockages at {oa["places"]} place(s) in {len(oa["nets"])} net(s)'
                  + (f' (e.g. {", ".join(oa["nets"][:3])})' if oa['nets'] else '')
                  + (f'; DEF BLOCKAGES: {len({b[3] for b in D.blockages})} on {", ".join(sorted({b[0] for b in D.blockages}))}' if D.blockages else ''))
    if D.drop:
        ev.append(f'already ignored for this net: OBS matching {", ".join(D.drop)}')
    if info is None:
        info = {}
    info.setdefault('obs', set())
    if pin is None:   # the net as a whole: list the pins without routing-layer shapes
        bad = [f'{i}/{p}' for i, p in D.nets[net]['conns'] if not [1 for l, _ in D.pin_shapes(i, p) if l in T.routing]]
        return 'LEF/DEF', f'{len(bad)} pin(s) of {net} have no shape on a routing layer ({", ".join(bad[:6])})', ev
    inst, pn = pin
    sh = D.pin_shapes(inst, pn)
    if not sh:
        if inst == 'PIN':
            return 'TESTCASE', f'I/O pin {pn} has no shape in the DEF PINS section', ev
        mac, loc = D.comps.get(inst, (None, None))
        M = T.macros.get(mac)
        if mac is None:
            return 'TESTCASE', f'instance {inst} is not in the DEF COMPONENTS', ev
        if not M:
            return 'LEF/DEF', f'cell {mac} of {inst} is not in the LEF files', ev
        return 'LEF/DEF', f'cell {mac} in the LEF has no shape for pin {pn} (LEF pins: {", ".join(sorted(M["pins"])[:8])})', ev
    b = (min(r[0] for _, r in sh), min(r[1] for _, r in sh), max(r[2] for _, r in sh), max(r[3] for _, r in sh))
    ev.append(f'pin {inst}/{pn}: {len(sh)} shape(s) on {"/".join(sorted({l for l, _ in sh}))}, '
              f'bbox ({um(b[0])}, {um(b[1])}) - ({um(b[2])}, {um(b[3])}) um')
    own = D.nets[net]['shapes']
    touched = any(_touch(a, c) for a in sh for c in own)
    obs = D.obstacles_labeled(net)
    shorts = sorted({o for a in sh for l, r, o in obs if l == a[0] and _rdist(a[1], r) == 0 and r[0] < a[1][2] and a[1][0] < r[2]
                     and r[1] < a[1][3] and a[1][1] < r[3] and o != f'{inst} OBS'})
    near = []
    for lay in [l for l in dict.fromkeys([a[0] for a in sh] + list(R.L[:2])) if l in T.routing]:
        Rl = T.routing[lay]; reach = int(round((Rl['spacing'] + Rl['width'] + Rl['pitch']) * u))
        ds = {}
        for l, r, o in obs:
            if l == lay:
                d = min(_rdist(a[1], r) for a in sh)
                if d <= reach and (o not in ds or d < ds[o]):
                    ds[o] = d
        near += [(d, f'{o} on {lay} {um(d)} um') for o, d in ds.items()]   # plan-view gap to the pin outline
    near = [t for _, t in sorted(near)[:6]]
    if near:
        ev.append('closest other shapes (plan-view gap to the pin; 0 = over or touching it): ' + '; '.join(near))
    # what keeps the via landings out over the pin, per layer
    ab = access_block(D, R, net, sh)
    info['blocked'] = max((x['blocked'] for x in ab), default=0.)
    pct = lambda v: f'{100 * v:.0f}%'
    for x in ab:
        ev.append(f'via landings over the pin on {x["layer"]} ({x["n"]} points): {pct(x["blocked"])} blocked'
                  + (' (' + ', '.join(f'{k} {pct(v)}' for k, v in sorted(x['groups'].items(), key=lambda kv: -kv[1])) + ')' if x['groups'] else '')
                  + (': ' + '; '.join(f'{o} {pct(s)}' for s, o, _, _ in x['owners'][:4]) if x['owners'] else '')
                  + (f'; {pct(x["under"])} of the pin lies inside cell OBS' if x['under'] else ''))
        for s, o, kd, i in x['owners']:
            if kd == 'obs':
                info['obs'].add((i, x['layer']))
            if kd == 'blockage':
                info.setdefault('blk', set()).add(i)
    ownov = sorted({o for c in own for l, r, o, kd, _, _ in D.obstacle_items(net) if kd in ('obs', 'blockage') and l == c[0]
                    and r[0] < c[1][2] and c[1][0] < r[2] and r[1] < c[1][3] and c[1][1] < r[3]})
    if ownov:
        ev.append(f'the original wiring of {net} itself overlaps {", ".join(ownov[:4])}: the input layout does not keep to these')
    if not touched:
        same = [(_rdist(a[1], c[1]), c[0]) for a in sh for c in own if a[0] == c[0]]
        dtxt = f'nearest wire of {net} on the same layer {um(min(same)[0])} um away on {min(same)[1]}' if same else \
            f'{net} has no wiring on the pin\'s layer(s)'
        alt, cell_miss, cell_tot = [], 0, 0
        mac, loc = D.comps.get(inst, (None, None)) if inst != 'PIN' else (None, None)
        M = T.macros.get(mac)
        if M and loc:   # every connection of this cell: how many does the wiring reach as placed, and turned otherwise?
            conns = [(n, p) for n, v in D.nets.items() for i, p in v['conns'] if i == inst and v['shapes']]
            reached = lambda o: sum(any(_touch(a, c) for a in D._place(M, M['pins'].get(p) or M['pins'].get(p.replace('\\', '')) or [],
                                                                        (loc[0], loc[1], o)) for c in D.nets[n]['shapes']) for n, p in conns)
            cell_tot = len(conns); now = reached(loc[2]); cell_miss = cell_tot - now
            alt = [o for o in ORIENTS if o != loc[2] and cell_tot and reached(o) == cell_tot]
            ev.append(f'cell {inst} ({mac}, {loc[2]}): the original wiring reaches {now} of its {cell_tot} connected pins')
        if tot and hit < .5 * tot:
            return 'LEF/DEF', (f'the original wiring does not reach this pin ({dtxt}), and the same holds for most pins of the design '
                               f'({tot - hit} of {tot}): the LEF cell geometry does not match the DEF, or the flow places the cells '
                               f'wrongly (ours)'), ev
        if alt:
            return 'LEF/DEF', (f'the original wiring reaches this pin only if {inst} is turned {"/".join(alt)} instead of {loc[2]} '
                               f'as the DEF says: the DEF orientation and the LEF disagree, or the flow\'s orientation handling is wrong (ours)'), ev
        if cell_tot > 1 and cell_miss == cell_tot:
            return 'TESTCASE', (f'none of the {cell_tot} connected pins of {inst} is reached by its wiring ({dtxt}): the cell is placed '
                                f'elsewhere in the DEF than where the wiring was drawn for it (or its LEF cell differs)'), ev
        return 'TESTCASE', (f'the original layout is open here: the original wiring of {net} does not reach this pin ({dtxt}); '
                            f'the pin is not connected in the input DEF either'), ev
    if shorts:
        return 'TESTCASE', f'the pin overlaps {", ".join(shorts[:4])} in the original layout (a short in the input)', ev
    wall = [x for x in ab if x['blocked'] >= .9]
    for kind, grp in (('obs', 'cell OBS (LEF)'), ('blockage', 'DEF blockage')):
        top = max(wall, key=lambda x: x['groups'].get(grp, 0), default=None)
        if not top or top['groups'].get(grp, 0) < .9:
            continue
        who = [f'{o} {pct(s)}' for s, o, kd, _ in top['owners'] if kd == kind][:3]
        under = max((x['under'] for x in ab), default=0)
        mine = [o for o in ownov if o.startswith(tuple(w.split(' ')[0] for w in who))]
        if kind == 'obs':
            return 'LEF/DEF', (f'the LEF OBS blocks the pin: on {top["layer"]} {pct(top["groups"][grp])} of the via landings over '
                               f'{inst}/{pn} are within spacing of cell OBS ({", ".join(who)})'
                               + (f', and {pct(under)} of the pin lies inside OBS' if under else '')
                               + (f'; the original wiring of {net} overlaps that OBS too, so the input layout itself does not keep to it'
                                  if mine else '')
                               + ': the OBS in the LEF covers the pin access (LEF definition, not a routing problem); '
                                 'the flow can ignore that OBS (--drop-obs)'), ev
        return 'TESTCASE', (f'a DEF routing blockage covers the pin: on {top["layer"]} {pct(top["groups"][grp])} of the via landings '
                            f'over {inst}/{pn} are within spacing of {", ".join(who)} (BLOCKAGES section of the input DEF); the flow can ignore it (--drop-obs)'), ev
    if wall:
        x = wall[0]
        return 'ROUTER', (f'the original wiring reaches this pin, but on {x["layer"]} {pct(x["blocked"])} of the via landings over it '
                          f'are within spacing of other shapes, mostly ' + ', '.join(f'{k} {pct(v)}' for k, v in
                                                                                  sorted(x['groups'].items(), key=lambda kv: -kv[1])[:2])
                          + ': the pin is reachable in the original layout only through spacing this router does not allow'), ev
    return 'ROUTER', (f'the original wiring reaches this pin and part of the pin is free, but this router finds no legal path '
                      f'to the free part (the closest shapes are listed below)'), ev


def _touch(a, b):
    return a[0] == b[0] and a[1][0] <= b[1][2] and b[1][0] <= a[1][2] and a[1][1] <= b[1][3] and b[1][1] <= a[1][3]


def split_terminals(D, R, name, min_pitches=3):
    """Partial rip-up: keep the net's original pin-access wiring, remove its long segments on the signal layers.
    Returns (terminals [((inst, pin), shapes)], kept pieces as DEF tuples, summary). The kept wiring is legal in the
    original layout, so the router only reconnects the parts that hold the pins (a dense pin area is never entered)."""
    T = D.tech; pcs = D.net_pieces(name); sig = set(R.L)
    pins = [((inst, pin), D.pin_shapes(inst, pin)) for inst, pin in D.nets[name]['conns']]
    pins = [p for p in pins if p[1]]
    long_ = lambda p: p['kind'] == 'seg' and p['layer'] in sig and p['length'] >= min_pitches * T.routing[p['layer']]['pitch'] * D.dbu
    keep = [p for p in pcs if p['kind'] != 'via' and not long_(p)]
    base = [sh for p in keep for sh in p['shapes']] + [sh for _, s in pins for sh in s]
    keep += [p for p in pcs if p['kind'] == 'via' and any(_touch(a, b) for a in p['shapes'] for b in base)]
    items = [('pin', k, s) for k, (_, s) in enumerate(pins)] + [('piece', k, p['shapes']) for k, p in enumerate(keep)]
    par = list(range(len(items)))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            if f(a) != f(b) and any(_touch(x, y) for x in items[a][2] for y in items[b][2]):
                par[f(a)] = f(b)
    comps = {}
    for k, it in enumerate(items):
        comps.setdefault(f(k), []).append(it)
    terms, kept = [], []
    for c in comps.values():
        ps = [pins[k] for kind, k, _ in c if kind == 'pin']
        if not ps:
            continue   # a piece of the old trunk that holds no pin: removed too
        terms.append((ps[0][0], [sh for kind, k, s in c for sh in s]))
        kept += [keep[k] for kind, k, _ in c if kind == 'piece']
    if len(terms) < 2:
        raise RuntimeError(f'{name}: nothing to reroute with the pin access kept ({len(terms)} connected part(s))')
    out = [(p['layer'], p['pts'], p.get('via') or (('RECT', p['rel']) if p['kind'] == 'rect' else None)) for p in kept]
    return terms, out, f'kept {len(kept)} of {len(pcs)} wiring pieces (those < {min_pitches} pitches), {len(terms)} parts to connect'


def route_one(D, R, name, g, mode, logl):
    """one net: full rip-up, else (auto) partial rip-up keeping the pin access -> (pieces, shapes, how)"""
    try:
        if mode == 'partial':
            raise RuntimeError('ROUTER_MODE=partial')
        pieces, shapes = R.route_net(name, g)
        return pieces, shapes, 'full rip-up'
    except RuntimeError as e1:
        if mode == 'full':
            raise
        errs = []; last = None; cands = [getattr(e1, 'pin', None), getattr(e1, 'also', None)]
        for mp in (3, 8, 20):   # keep the short pieces first; if that fails, keep longer ones (less is rerouted)
            try:
                terms, kept, how = split_terminals(D, R, name, mp)
                pieces, shapes = R.route_net(name, g, terms=terms)
                break
            except RuntimeError as e2:
                errs.append(f'keeping pieces < {mp} pitches: {e2}'); last = e2 if getattr(e2, 'pin', None) else last
                cands += [getattr(e2, 'pin', None), getattr(e2, 'also', None)]
        else:
            src = e1 if getattr(e1, 'pin', None) else last
            e = NoPath(f'{e1} | ' + ' | '.join(errs), getattr(src, 'pin', None), getattr(src, 'also', None))
            e.cands = [c for c in dict.fromkeys(cands) if c]   # every pin an attempt stopped at
            raise e
        logl.append(f'FLOW {name}: full rip-up not possible ({e1}); {how}')
        return kept + pieces, shapes, how


RANK = {'TESTCASE': 0, 'LEF/DEF': 0, 'ROUTER': 1}


def diagnose_pins(D, R, name, e):
    """diagnose the pins the attempts (full and partial rip-up) could not reach or started from; the one with an
    input-side cause (else the most blocked one) is reported -> (verdict, why, evidence, pin, info)"""
    best = None
    for pin in list(dict.fromkeys([e.pin, getattr(e, 'also', None)] + getattr(e, 'cands', [])))[:6]:
        if pin is None and best is not None:
            continue
        info = {}
        v, w, ev = diagnose(D, R, name, pin, info)
        if pin is not None and pin != e.pin:
            w = f'(where {"an attempt" if pin != getattr(e, "also", None) else "the path"} stopped or started) {w}'
        key = (RANK.get(v, 1), -info.get('blocked', 0.), -len(info.get('obs', ())))
        if best is None or key < best[0]:
            best = (key, v, w, ev, pin, info)
    return best[1:]


def run(lefs, def_in, def_out, nets, layers, guide_file=None, log=None, drc=None, guide_out=None):
    t0 = time.time()
    T = Tech(lefs)
    D = Design(def_in, T)
    lo, hi = layers.split('-')
    rl = T.rorder
    sig = rl[rl.index(lo):rl.index(hi) + 1]
    mx = int(__import__('os').environ.get('ROUTER_MAX_LAYERS', '0'))   # 0: every signal layer
    sig = sig[:mx] if mx > 0 else sig
    guides = {}
    if guide_file and Path(guide_file).exists():
        for m in re.finditer(r'^(\S+)\s*\n\(\s*\n(.*?)^\)', Path(guide_file).read_text(), re.M | re.S):
            guides[m.group(1)] = [tuple(int(v) for v in ln.split()[:4]) for ln in m.group(2).splitlines() if ln.strip()]
    R = Router(T, D, sig)
    text = D.text; logl = [f'FLOW builtin router: layers {"-".join([sig[0], sig[-1]])}, grid pitch {R.P} dbu']
    viol = []
    # GCELLGRID the flow reads (from the DEF, else 15 tracks)
    x1, y1, x2, y2 = D.die
    gs = 15 * R.P
    gx = D.gcell.get('X', (x1, (x2 - x1) // gs + 1, gs)); gy = D.gcell.get('Y', (y1, (y2 - y1) // gs + 1, gs))
    logl += [f'GCELLGRID X {gx[0]} DO {gx[1]} STEP {gx[2]}', f'GCELLGRID Y {gy[0]} DO {gy[1]} STEP {gy[2]}']
    gtxt = []
    edits = []
    spec = __import__('os').environ.get('ROUTER_DROP_OBS', 'auto').strip()   # auto | off | spec,spec (see Design.drop_match)
    parts = [x for x in re.split(r'[,\s]+', spec) if x]
    drop_auto = 'auto' in parts
    fixed_drop = [x for x in parts if x not in ('auto', 'off', 'none')]
    if fixed_drop:
        logl.append(f'FLOW ignoring OBS for every net: {", ".join(fixed_drop)}')
    for net in nets:
        name = net if net in D.nets else next((n for n in D.nets if n.replace('\\', '') == net), None)
        if name is None:
            raise RuntimeError(f'FLOW_ERROR net not found: {net}')
        g = guides.get(net) or guides.get(name)
        mode = __import__('os').environ.get('ROUTER_MODE', 'auto')   # auto: full rip-up, else keep the pin access
        D.drop = list(fixed_drop)
        tried = []   # (drop specs, failure diagnosis) of the attempts before the one that routed
        while True:
            try:
                pieces, shapes, how = route_one(D, R, name, g, mode, logl)
                break
            except NoPath as e:
                verdict, why, ev, pin, info = diagnose_pins(D, R, name, e)
                ncap, natt = str(e).count('[SEARCH LIMIT'), str(e).count(' | ') + 1
                lim = getattr(R, 'max_expand', 0)
                if ncap:
                    ev = [f'search limit: {ncap} of {natt} attempt(s) stopped at the node limit ({lim} nodes) without a '
                          f'path; such an attempt says nothing about the pin'] + ev
                if ncap == natt:   # every attempt gave up: the pin evidence below may be real, but it is not why these stopped
                    why = (f'every attempt stopped at the search limit ({lim} expanded nodes) before it found a path, so '
                           f'this is the router giving up, not a proven block (pin evidence: {verdict}: {why})')
                    verdict = 'ROUTER'
                e.diag = [f'DIAG {name} pin {"/".join(pin) if pin else "-"}: {verdict}: {why}'] + [f'DIAG   {x}' for x in ev]
                new = [f'inst={i}:{l}' for i, l in sorted(info['obs'])] + sorted(info.get('blk', ()))
                new = [s for s in new if s not in D.drop]
                e.diag += [f'DIAG   before: ignoring {", ".join(d) or "nothing"} also failed: {w}' for d, w in tried]
                if not drop_auto or len(tried) >= 6 or not (verdict == 'ROUTER' or 'LEF OBS' in why or 'DEF routing blockage' in why):
                    raise   # an input-side cause that ignoring OBS cannot fix (open, short, orientation, missing cell)
                if not new:   # no OBS / blockage over the pins left to blame: ignoring more would not help
                    raise
                tried.append((list(D.drop), e.diag[0][5:]))
                D.drop += new
        if D.drop:
            hit = [it for it in D.obstacle_items(name) if D.drop_match(it, name)]
            ov = sp_ = 0; who = {}
            for l, r in shapes:
                Rl = T.routing.get(l); s_ = int(round(Rl['spacing'] * D.dbu)) if Rl else 0
                for it in hit:
                    o = it[1]
                    if it[0] == l and r[0] < o[2] + s_ and o[0] < r[2] + s_ and r[1] < o[3] + s_ and o[1] < r[3] + s_:
                        if r[0] < o[2] and o[0] < r[2] and r[1] < o[3] and o[1] < r[3]:
                            ov += 1
                        else:
                            sp_ += 1
                        who[f'{it[2]} on {l}'] = 1
            logl.append(f'FLOW relaxation {name}: routed while ignoring {", ".join(D.drop)} '
                        f'({len(hit)} OBS / blockage shape(s)); the new wiring overlaps them at {ov} place(s) and is within spacing of them '
                        f'at {sp_}' + (f' ({", ".join(list(who)[:6])})' if who else ''))
            logl += [f'RELAX   attempt {k + 1} (ignoring {", ".join(d) or "nothing"}) failed: {why}' for k, (d, why) in enumerate(tried)]
        viol += self_check(shapes, D.obstacles(name), T, D.dbu, name)
        s, e = D.nets[name]['span']
        st = text[s:e]
        head = st.split('+')[0].rstrip().rstrip(';').rstrip()
        use = f" + USE {D.nets[name]['use']}" if D.nets[name]['use'] else ''
        edits.append((s, e, f'{head}\n      {def_wiring(pieces)}{use} ;'))
        logl.append(f'FLOW routed {name} ({how}): {len(pieces)} pieces, {sum(1 for p in pieces if isinstance(p[2], str))} vias')
        # pin-access boxes (layers below the signal range) for the flow's guides
        boxes = [(l, r) for inst, pin in D.nets[name]['conns'] for l, r in D.pin_shapes(inst, pin)
                 if l in T.routing and T.rorder.index(l) < T.rorder.index(sig[0])]
        gtxt.append(f'{name}\n(\n' + ''.join(f'{r[0]} {r[1]} {r[2]} {r[3]} {l}\n' for l, r in boxes) + ')\n')
    for s, e, new in sorted(edits, reverse=True):
        text = text[:s] + new + text[e:]
    Path(def_out).write_text(text)
    if drc:
        Path(drc).write_text('\n'.join(viol) + ('\n' if viol else ''))
    if guide_out:
        Path(guide_out).write_text(''.join(gtxt))
    logl.append(f'FLOW builtin router done in {time.time() - t0:.1f} s, {len(viol)} self-check violation(s)')
    logl.append('FLOW_DONE')
    if log:
        Path(log).write_text('\n'.join(logl) + '\n')
    return viol


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--lefs', required=True); ap.add_argument('--def', dest='def_in', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--nets', required=True); ap.add_argument('--layers', required=True)
    ap.add_argument('--guide'); ap.add_argument('--log'); ap.add_argument('--drc'); ap.add_argument('--guide-out')
    a = ap.parse_args()
    try:
        run(a.lefs.split(','), a.def_in, a.out, a.nets.split(), a.layers, a.guide, a.log, a.drc, a.guide_out)
    except Exception as e:   # noqa: BLE001
        if a.log:
            Path(a.log).write_text(f'Error: builtin router: {e}\n' + ''.join(x + '\n' for x in getattr(e, 'diag', [])))
        print(f'Error: builtin router: {e}', file=sys.stderr)
        for x in getattr(e, 'diag', []):
            print(x, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
