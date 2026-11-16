"""Prompt features for one candidate tree against the current routed layout (LLM v2 representation).
Copied without behavioural change from the research code (features_v2.py) so the prompt text matches the shipped
adapters exactly; only the import of the coupling proxy changed. setup_units() adapts units to the design."""
from collections import defaultdict
from . import proxy as pm

DBU = pm.DBU
TOL_PITCH = 2.0          # candidate centreline vs routed track tolerance, in pitches of that layer
LAYERS = sorted(pm.PITCH_UM)


def runs(shapes):
    return pm.wire_runs(shapes)          # {(layer,'H'|'V'): [(track_lo, track_hi, a, b)]}


def segs(geometry):
    out = []
    for x1, y1, x2, y2 in geometry:
        if y1 == y2 and x1 != x2:
            out.append(('H', y1, min(x1, x2), max(x1, x2)))
        elif x1 == x2 and y1 != y2:
            out.append(('V', x1, min(y1, y2), max(y1, y2)))
    return out


def mirror(geometry, axis):
    return [[2 * axis - x1, y1, 2 * axis - x2, y2] for x1, y1, x2, y2 in geometry]


def union_len(iv):
    iv = sorted(iv); tot = 0; cur = None
    for a, b in iv:
        if cur is None or a > cur[1]:
            if cur: tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur: tot += cur[1] - cur[0]
    return tot


def covered(a, b, cover):
    """length of [a,b] covered by intervals in cover"""
    return union_len([(max(a, x), min(b, y)) for x, y in cover if min(b, y) > max(a, x)])


def near_cover(o, track_lo, track_hi, layer, cand):
    """candidate spans of orientation o whose centreline lies within tolerance of the routed track band"""
    tol = TOL_PITCH * pm.PITCH_UM[layer] * DBU
    return [(a, b) for oo, t, a, b in cand if oo == o and track_lo - tol <= t <= track_hi + tol]


def coupled_intervals(own_runs, other_runs):
    """[(layer, o, own_run, (a,b) coupled span)] using the plan_metrics coupling rule"""
    out = []
    for key, rr in own_runs.items():
        layer, o = key
        gmax = pm.K * pm.PITCH_UM[layer] * DBU
        for lo1, hi1, a1, b1 in rr:
            for lo2, hi2, a2, b2 in other_runs.get(key, []):
                gap = max(lo2 - hi1, lo1 - hi2)
                if gap <= 0 or gap > gmax:
                    continue
                a, b = max(a1, a2), min(b1, b2)
                if b > a:
                    out.append((layer, o, (lo1, hi1, a1, b1), (a, b)))
    return out


def analyse(nets, pairs, axis, cand, current_metrics):
    """nets: {net: shapes}; pairs: set of frozenset; cand: pool candidate dict. Returns a feature dict."""
    net, partner = cand['net'], cand.get('partner') or ''
    own = [net] + ([partner] if partner else [])
    geometry = {net: cand['imported']['geometry']}
    if partner:
        geometry[partner] = mirror(cand['imported']['geometry'], axis)
    elif cand.get('self_sym'):
        geometry[net] = geometry[net] + mirror(geometry[net], axis)
    signals = [n for n in nets if not pm.is_power(n)]
    others = [n for n in signals if n not in own]
    other_runs = {o: runs(nets[o]) for o in others}
    per_neighbor = defaultdict(lambda: dict(current=0., removed=0., new_same_layer=0., new_any_layer=0.))
    own_len = new_len = cand_len = kept_len = 0.
    layers_used = defaultdict(float)
    for n in own:
        orun = runs(nets.get(n, []))
        cs = segs(geometry[n])
        cand_len += sum(b - a for _, _, a, b in cs)
        for (layer, o), rr in orun.items():
            for lo, hi, a, b in rr:
                own_len += b - a; layers_used[(layer, o)] += b - a
                kept_len += covered(a, b, near_cover(o, lo, hi, layer, cs))
        # current coupling and the part that sits on abandoned path
        for o_net in others:
            if frozenset([n, o_net]) in pairs:
                continue
            for layer, o, (lo, hi, a1, b1), (a, b) in coupled_intervals(orun, other_runs[o_net]):
                c = (b - a)
                k = covered(a, b, near_cover(o, lo, hi, layer, cs))
                per_neighbor[o_net]['current'] += c
                per_neighbor[o_net]['removed'] += c - k
        # new path: candidate spans not near any routed track of the same net/orientation
        used_layers = {o: {l for (l, oo) in layers_used if oo == o} for o in 'HV'}
        for o, t, a, b in cs:
            cover = [(ra, rb) for (layer, oo), rr in orun.items() if oo == o for lo, hi, ra, rb in rr
                     if lo - TOL_PITCH * pm.PITCH_UM[layer] * DBU <= t <= hi + TOL_PITCH * pm.PITCH_UM[layer] * DBU]
            fresh = [(a, b)]
            for x, y in sorted(cover):
                nxt = []
                for p, q in fresh:
                    if y <= p or x >= q: nxt.append((p, q)); continue
                    if x > p: nxt.append((p, x))
                    if y < q: nxt.append((y, q))
                fresh = nxt
            for p, q in fresh:
                new_len += q - p
                for o_net in others:
                    if frozenset([n, o_net]) in pairs:
                        continue
                    for (layer, oo), rr in other_runs[o_net].items():
                        if oo != o:
                            continue
                        gmax = pm.K * pm.PITCH_UM[layer] * DBU
                        for lo, hi, ra, rb in rr:
                            d = max(lo - t, t - hi)
                            if d > gmax or d < -gmax:
                                continue
                            ov = min(q, rb) - max(p, ra)
                            if ov <= 0:
                                continue
                            per_neighbor[o_net]['new_any_layer'] += ov
                            if layer in used_layers[o] or not used_layers[o]:
                                per_neighbor[o_net]['new_same_layer'] += ov
    nb = {k: {kk: vv / DBU for kk, vv in v.items()} for k, v in per_neighbor.items()}
    cur = sum(v['current'] for v in nb.values()); rem = sum(v['removed'] for v in nb.values())
    new_same = sum(v['new_same_layer'] for v in nb.values()); new_any = sum(v['new_any_layer'] for v in nb.values())
    return dict(net=net, partner=partner, self_sym=bool(cand.get('self_sym')), pins=cand['imported']['original_pin_count'],
                segments=len(segs(cand['imported']['geometry'])), cand_len_um=cand_len / DBU, own_len_um=own_len / DBU,
                kept_frac=kept_len / own_len if own_len else 0., new_len_um=new_len / DBU,
                current_cpl_um=cur, removed_cpl_um=rem, new_same_layer_um=new_same, new_any_layer_um=new_any,
                heuristic_delta_c_um=new_same - rem, neighbors=nb, block=current_metrics,
                layers={f'M{l // 2}{o}': v / DBU for (l, o), v in layers_used.items()})


def fmt_neighbors(nb, key, limit=6):
    items = sorted(((v[key], k) for k, v in nb.items() if v[key] > 0.05), reverse=True)[:limit]
    return ', '.join(f'{k} {v:.1f}' for v, k in items) or 'none'


def render(f, bench, ckt, feedback=None):
    b = f['block']
    lines = [
        'Task: decide whether rerouting one net with the proposed tree improves layout utility '
        'U = coupling reduction - 0.25*WL increase - 0.05*via increase (all relative to the original layout). '
        'Coupling = same-layer parallel run length to other nets within 2 pitches (um).',
        f'Circuit {bench}, block {ckt}: coupling {b["C"]:.2f} um, wirelength {b["WL"]:.1f} um, vias {b["vias"]}.',
        f'Target net {f["net"]}: {f["pins"]} pins'
        + (f', symmetric partner {f["partner"]} is rerouted with the mirrored tree' if f['partner'] else '')
        + (', self-symmetric' if f['self_sym'] else '') + '.',
        f'Current route: {f["own_len_um"]:.1f} um of wire on {", ".join(f"{k} {v:.0f}" for k, v in sorted(f["layers"].items())) or "no layers"}; '
        f'coupling to other nets {f["current_cpl_um"]:.2f} um ({fmt_neighbors(f["neighbors"], "current")}).',
        f'Proposed tree: {f["segments"]} segments, {f["cand_len_um"]:.1f} um centreline; keeps {100 * f["kept_frac"]:.0f}% of the current path, '
        f'adds {f["new_len_um"]:.1f} um of new path.',
        f'Coupling on abandoned path: {f["removed_cpl_um"]:.2f} um ({fmt_neighbors(f["neighbors"], "removed")}).',
        f'New path runs parallel to other nets on layers this net uses: {f["new_same_layer_um"]:.2f} um ({fmt_neighbors(f["neighbors"], "new_same_layer")}); '
        f'on any layer: {f["new_any_layer_um"]:.2f} um.',
    ]
    if feedback:
        lines.append('Measured results of earlier reroutes in this layout:')
        for r in feedback:
            lines.append(f'- net {r["net"]}: removed {r["removed"]:.1f} um, new parallel {r["new"]:.1f} um -> '
                         f'coupling {r["dC"]:+.2f} um, WL {r["dWL"]:+.1f} um, vias {r["dV"]:+d}, U {r["dU"]:+.4f}.')
    lines.append('Will this reroute improve U? Answer Yes or No.')
    return '\n'.join(lines)


def setup_units(geo):
    """Point the proxy and this module at the design's DBU and routing-layer pitches (layer key = 2*routing level)."""
    global DBU, LAYERS
    pm.DBU = DBU = geo['dbu']; pm.MIN_LEN = geo['dbu']
    pm.PITCH_UM = {2 * l['level']: l['pitch'] / geo['dbu'] for l in geo['layers'].values()}
    LAYERS = sorted(pm.PITCH_UM)
