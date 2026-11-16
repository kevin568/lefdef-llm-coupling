"""Geometric coupling proxy used in the LLM prompt (same-layer parallel run length within K pitches).
Copied without behavioural change from the research code (plan_metrics.wire_runs / coupling / is_power), so prompts
are byte-identical to the ones the shipped adapters were trained and evaluated on. Units are set per design by
features.setup_units()."""
from collections import defaultdict

DBU = 2000
PITCH_UM = {}        # {2*routing_level: pitch_um}
K = 2.0
MIN_LEN = 1 * DBU
ASPECT = 3.0
POWER = {'vdda', 'vddd', 'vssa', 'vssd', 'vdd', 'vss', 'gnd', 'VDD', 'VSS', 'GND'}


def is_power(n):
    return n in POWER or n.lower() in {'vdd', 'vss', 'gnd', 'vdda', 'vddd', 'vssa', 'vssd'}


def wire_runs(shapes):
    """{(layer, 'H'|'V'): [(track_lo, track_hi, a, b)]} merged collinear runs of one net."""
    groups = defaultdict(list)
    for layer, dt, r in shapes:
        if dt != 0 or layer not in PITCH_UM:
            continue
        w, h = r[2] - r[0], r[3] - r[1]
        if w >= h:
            if w < MIN_LEN or w < ASPECT * max(h, 1):
                continue
            groups[(layer, 'H', r[1], r[3])].append((r[0], r[2]))
        else:
            if h < MIN_LEN or h < ASPECT * max(w, 1):
                continue
            groups[(layer, 'V', r[0], r[2])].append((r[1], r[3]))
    out = defaultdict(list)
    for (layer, o, lo, hi), iv in groups.items():
        iv.sort()
        cur = list(iv[0])
        for a, b in iv[1:]:
            if a <= cur[1]:
                cur[1] = max(cur[1], b)
            else:
                out[(layer, o)].append((lo, hi, cur[0], cur[1]))
                cur = [a, b]
        out[(layer, o)].append((lo, hi, cur[0], cur[1]))
    return out


def coupling(nets, sym_nets, pair_of=None):
    """(total um, um involving a symmetric-pair net, {net: um}, um between the two nets of one symmetric pair)"""
    by_lo = defaultdict(list)
    for net, shapes in nets.items():
        if is_power(net):
            continue
        for key, runs in wire_runs(shapes).items():
            for lo, hi, a, b in runs:
                by_lo[key].append((lo, hi, a, b, net))
    tot = sym = intra = 0
    pair_of = pair_of or {}
    per = defaultdict(int)
    for (layer, o), runs in by_lo.items():
        gmax = K * PITCH_UM[layer] * DBU
        runs.sort()
        for i, (lo1, hi1, a1, b1, n1) in enumerate(runs):
            for lo2, hi2, a2, b2, n2 in runs[i + 1:]:
                if lo2 - hi1 > gmax:
                    break
                if n1 == n2:
                    continue
                gap = max(lo2 - hi1, lo1 - hi2)
                if gap <= 0 or gap > gmax:
                    continue
                ov = min(b1, b2) - max(a1, a2)
                if ov <= 0:
                    continue
                tot += ov
                per[n1] += ov
                per[n2] += ov
                if n1 in sym_nets or n2 in sym_nets:
                    sym += ov
                if pair_of.get(n1) == n2:
                    intra += ov
    return tot / DBU, sym / DBU, {k: v / DBU for k, v in per.items()}, intra / DBU
