"""SPEF reader for coupling metrics.
total_cc  : sum of all coupling capacitors, each counted once (fF)
per_net_cc: coupling attributed to a net, crediting each coupling capacitor to BOTH of its nets (fF)
listed_cc : coupling listed inside a net's own *D_NET section (what the research pool selection used)
total_cap : *D_NET total capacitance per net (fF)"""
from collections import defaultdict
from pathlib import Path


def read(path):
    names, unit = {}, 1e3
    pairs, listed, total_cap, pin_owner = {}, defaultdict(float), {}, {}
    cur, section = None, None
    for ln in Path(path).read_text(errors='replace').splitlines():
        if ln.startswith('*C_UNIT'):
            v, u = ln.split()[1:3]; unit = float(v) * {'PF': 1e3, 'FF': 1.0}[u.upper()]
        elif cur is None and ln.startswith('*') and len(ln.split()) == 2 and ln.split()[0][1:].isdigit():
            k, v = ln.split(); names[k] = v
        elif ln.startswith('*D_NET'):
            _, k, tot = ln.split()[:3]; cur = names.get(k, k); total_cap[cur] = float(tot) * unit; section = None
        elif ln.startswith('*CONN'): section = 'conn'
        elif ln.startswith('*CAP'): section = 'cap'
        elif ln.startswith('*RES') or ln.startswith('*END'): section = None
        elif section == 'conn' and (ln.startswith('*I') or ln.startswith('*P')):
            pin_owner[ln.split()[1]] = cur
        elif section == 'cap':
            f = ln.split()
            if len(f) == 4:
                v = float(f[3]) * unit
                pairs[frozenset([f[1], f[2]])] = v
                listed[cur] += v

    def owner(node):
        if node in pin_owner: return pin_owner[node]
        k = node.split(':')[0]
        return names.get(k, k)

    per = defaultdict(float)
    for key, v in pairs.items():
        ab = list(key)
        if len(ab) == 2:
            for n in {owner(ab[0]), owner(ab[1])}:
                per[n] += v
    return dict(total_cc=sum(pairs.values()), per_net_cc=dict(per), listed_cc=dict(listed), total_cap=total_cap,
                n_coupling_caps=len(pairs))
