"""Candidate Steiner-like trees per net (combs, MST L-variants, Hanan trees), seeded by the net name.
Copied without behavioural change from the research code so the candidate set is reproducible."""
import hashlib
from collections import defaultdict
import numpy as np


def merge(segs):
    groups = defaultdict(list)
    for x1, y1, x2, y2 in segs:
        if (x1, y1) == (x2, y2): continue
        if y1 == y2: groups[('H', y1)].append(sorted([x1, x2]))
        else:
            assert x1 == x2; groups[('V', x1)].append(sorted([y1, y2]))
    out = []
    for (o, t), ivs in sorted(groups.items()):
        merged = []
        for a, b in sorted(ivs):
            if merged and a <= merged[-1][1]: merged[-1][1] = max(b, merged[-1][1])
            else: merged.append([a, b])
        out += [[a, t, b, t] if o == 'H' else [t, a, t, b] for a, b in merged]
    return out


def lshape(p, q, hfirst):
    bend = (q[0], p[1]) if hfirst else (p[0], q[1])
    return [[*p, *bend], [*bend, *q]]


def covers(geom, pins):
    return all(any((y == y1 == y2 and min(x1, x2) <= x <= max(x1, x2)) or (x == x1 == x2 and min(y1, y2) <= y <= max(y1, y2))
                   for x1, y1, x2, y2 in geom) for x, y in pins)


def comb(pins, mode):
    root = pins[0]
    return merge([s for p in pins[1:] for s in lshape(root, p, mode == 'x_first')])


def mst_edges(pins):
    n = len(pins); used = {0}; edges = []
    while len(used) < n:
        d, a, b = min((abs(pins[i][0] - pins[j][0]) + abs(pins[i][1] - pins[j][1]), i, j) for i in used for j in range(n) if j not in used)
        edges.append((a, b)); used.add(b)
    return edges


def hanan(points, rng):
    xs = sorted({p[0] for p in points}); ys = sorted({p[1] for p in points}); coords = [(x, y) for x in xs for y in ys]
    ids = {p: i for i, p in enumerate(coords)}; ny = len(ys); edges = []; factor = float(np.exp(rng.uniform(-.7, .7)))
    for i, (x, y) in enumerate(coords):
        for j in ([i + ny] if i + ny < len(coords) else []) + ([i + 1] if i % ny + 1 < ny else []):
            X, Y = coords[j]; edges.append(((abs(X - x) * factor + abs(Y - y) / factor) * float(np.exp(rng.uniform(-1.4, 1.4))), i, j))
    parent = list(range(len(coords)))
    def root(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    adj = [set() for _ in coords]
    for _, a, b in sorted(edges):
        aa, bb = root(a), root(b)
        if aa != bb: parent[aa] = bb; adj[a].add(b); adj[b].add(a)
    terminals = {ids[tuple(p)] for p in points}
    todo = [i for i, a in enumerate(adj) if len(a) == 1 and i not in terminals]; removed = set()
    while todo:
        a = todo.pop()
        if a in removed: continue
        removed.add(a)
        for b in list(adj[a]):
            adj[b].remove(a)
            if len(adj[b]) == 1 and b not in terminals: todo.append(b)
        adj[a].clear()
    return merge([[*coords[a], *coords[b]] for a in range(len(coords)) for b in adj[a] if a < b])


def zshape(p, q, hfirst, f):
    """p -> q with one jog at fraction f of the way (horizontal first: along x to the jog, then y, then x)"""
    if hfirst:
        xm = int(round(p[0] + f * (q[0] - p[0])))
        return [s for s in ([p[0], p[1], xm, p[1]], [xm, p[1], xm, q[1]], [xm, q[1], q[0], q[1]]) if s[:2] != s[2:]]
    ym = int(round(p[1] + f * (q[1] - p[1])))
    return [s for s in ([p[0], p[1], p[0], ym], [p[0], ym, q[0], ym], [q[0], ym, q[0], q[1]]) if s[:2] != s[2:]]


def ushape(p, q, hfirst, d):
    """p -> q with the trunk moved sideways by d (a detour away from what runs next to the direct path)"""
    if hfirst:
        y = min(p[1], q[1]) + d if d < 0 else max(p[1], q[1]) + d
        return [s for s in ([p[0], p[1], p[0], y], [p[0], y, q[0], y], [q[0], y, q[0], q[1]]) if s[:2] != s[2:]]
    x = min(p[0], q[0]) + d if d < 0 else max(p[0], q[0]) + d
    return [s for s in ([p[0], p[1], x, p[1]], [x, p[1], x, q[1]], [x, q[1], q[0], q[1]]) if s[:2] != s[2:]]


def pool_for_net(net, pins, extra=0):
    """candidate trees of a net: x_first / y_first combs, 8 MST and 8 Hanan variants (fixed recipe), and with
    extra > 0 that many Z-shaped trees (each MST edge jogs at a random point) and extra // 2 detour trees (the
    trunk of each MST edge moved sideways by 10-40 % of the net's extent), from a separate seed so the first ones
    stay the same"""
    pins = sorted(set(tuple(p) for p in pins))
    # the 'WU119:' prefix is part of the seed recipe; changing it changes every candidate set
    seed = int(hashlib.sha256(f'WU119:{net}'.encode()).hexdigest()[:8], 16); rng = np.random.default_rng(seed)
    opts = [('x_first', comb(pins, 'x_first')), ('y_first', comb(pins, 'y_first'))]
    edges = mst_edges(pins)
    for t in range(8):
        dirs = rng.integers(0, 2, len(edges)).astype(bool)
        opts.append((f'mst{t}', merge([s for k, (a, b) in enumerate(edges) for s in lshape(pins[a], pins[b], dirs[k])])))
    for t in range(8):
        opts.append((f'hanan{t}', hanan(pins, rng)))
    if extra > 0 and len(pins) >= 2:
        r2 = np.random.default_rng(int(hashlib.sha256(f'WU119x:{net}'.encode()).hexdigest()[:8], 16))
        span = max(max(p[0] for p in pins) - min(p[0] for p in pins), max(p[1] for p in pins) - min(p[1] for p in pins), 1)
        for t in range(extra):
            dirs = r2.integers(0, 2, len(edges)).astype(bool); fr = r2.uniform(.2, .8, len(edges))
            opts.append((f'z{t}', merge([s for k, (a, b) in enumerate(edges) for s in zshape(pins[a], pins[b], dirs[k], fr[k])])))
        for t in range(extra // 2):
            dirs = r2.integers(0, 2, len(edges)).astype(bool)
            ds = (r2.uniform(.1, .4, len(edges)) * span * r2.choice([-1, 1], len(edges))).astype(int)
            opts.append((f'detour{t}', merge([s for k, (a, b) in enumerate(edges) for s in ushape(pins[a], pins[b], dirs[k], int(ds[k]))])))
    seen = set(); out = []
    for kind, g in opts:
        key = tuple(map(tuple, g))
        if key in seen or not covers(g, pins): continue
        seen.add(key); out.append(dict(candidate_id=f'{net}:{kind}', net=net, partner=None, self_sym=False, kind=kind,
                                       imported=dict(original_pin_count=len(pins), geometry=g)))
    return out
