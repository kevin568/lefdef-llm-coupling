"""Coupling without an OpenRCX rules file: the geometric proxy written as a SPEF, so every later stage is unchanged.

Coupling between two nets = the length (um) their wires run parallel on the same routing layer within K (=2) routing
pitches (wire runs >= 1 um, aspect >= 3), the rule of proxy.py. Each pair is written as one coupling "capacitor" whose
value is that length; the SPEF says *C_UNIT 1 FF, so the numbers are read as fF but mean um of parallel run.
Ranking of nets by this proxy vs OpenRCX coupling: Spearman 0.90-0.96 on 6 public layouts (SKY130, Nangate45, IHP)."""
import re
from collections import defaultdict

K = 2.0


def _runs(shapes, pitch, dbu):
    """{(layer, 'H'|'V'): [(track_lo, track_hi, a, b)]} merged collinear runs of one net (proxy.wire_runs rule)"""
    groups = defaultdict(list)
    for s in shapes:
        layer, r = s['layer'], s['rect']
        if layer not in pitch:
            continue
        w, h = r[2] - r[0], r[3] - r[1]
        if w >= h:
            if w < dbu or w < 3.0 * max(h, 1):
                continue
            groups[(layer, 'H', r[1], r[3])].append((r[0], r[2]))
        else:
            if h < dbu or h < 3.0 * max(w, 1):
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


def pairs(geo):
    """{(net1, net2): um} for signal nets (power/ground nets are not in geo)"""
    dbu = geo['dbu']
    pitch = {name: l['pitch'] / dbu for name, l in geo['layers'].items() if l.get('pitch')}
    by = defaultdict(list)
    for net, v in geo['nets'].items():   # geo has no power/ground nets; clock nets count as neighbours (as in OpenRCX)
        for key, rr in _runs(v['shapes'], pitch, dbu).items():
            for lo, hi, a, b in rr:
                by[key].append((lo, hi, a, b, net))
    out = defaultdict(int)
    for (layer, o), rr in by.items():
        g = K * pitch[layer] * dbu
        rr.sort()
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
                if ov > 0:
                    out[tuple(sorted((n1, n2)))] += ov
    return {k: v / float(dbu) for k, v in out.items()}


def write_spef(geo, path):
    """Minimal SPEF: name map, one *D_NET per net with its coupling entries (listed in both nets' sections)."""
    pr = pairs(geo)
    nets = sorted(geo['nets'])
    idx = {n: i + 1 for i, n in enumerate(nets)}
    per = defaultdict(list)
    for (a, b), um in pr.items():
        per[a].append((a, b, um)); per[b].append((a, b, um))
    L = ['*SPEF "IEEE 1481-1998"', '*DESIGN "%s"' % geo.get('design', ''), '*PROGRAM "lefdef-llm-coupling proxy"',
         '*DIVIDER /', '*DELIMITER :', '*BUS_DELIMITER [ ]', '*T_UNIT 1 NS', '*C_UNIT 1 FF', '*R_UNIT 1 OHM',
         '*L_UNIT 1 HENRY', '', '*NAME_MAP']
    L += ['*%d %s' % (idx[n], n) for n in nets]
    L.append('')
    for n in nets:
        caps = per.get(n, [])
        L += ['*D_NET *%d %.6f' % (idx[n], sum(um for _, _, um in caps)), '*CAP']
        L += ['%d *%d:1 *%d:1 %.6f' % (k + 1, idx[a], idx[b], um) for k, (a, b, um) in enumerate(caps)]
        L += ['*END', '']
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    return sum(pr.values())
