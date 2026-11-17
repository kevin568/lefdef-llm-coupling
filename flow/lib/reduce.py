"""Reduced copies of a (repaired) design for locating what the detailed router cannot handle.

Each reduction leaves one kind of content out of a copy of the DEF or of the cell LEFs. Running the router probe on
each copy shows which content makes the router fail (tools/diagnose.py check 11), and the flow can keep running with
the least destructive reduction the router accepts (lib/common.py). Original files are never changed."""
import re
from pathlib import Path

from . import common as C
from . import preflight as P


def _section_span(text, name):
    m = re.search(r'^[ \t]*' + name + r'[ \t]+\d+[ \t]*;[^\n]*\n', text, re.M)
    if not m:
        return None
    e = re.search(r'^[ \t]*END[ \t]+' + name + r'\b[^\n]*\n?', text[m.end():], re.M)
    return (m.start(), m.end(), m.end() + e.start(), m.end() + e.end()) if e else None


def _rewrite(text, name, fn):
    """fn(statement) -> statement or None for each '- ...;' statement of a section; the count is updated"""
    sp = _section_span(text, name)
    if not sp:
        return text
    hs, he, bs, be = sp
    stmts = re.split(r'(?<=;)[ \t]*\r?\n', text[he:bs])
    kept = [x for x in (fn(s) if s.strip().startswith('-') else s for s in stmts) if x is not None]
    n = sum(1 for s in kept if s.strip().startswith('-'))
    head = re.sub(r'\d+', str(n), text[hs:he], count=1)
    return text[:hs] + head + '\n'.join(k for k in kept if k.strip()) + '\n' + text[bs:]


def _drop_section(text, name):
    sp = _section_span(text, name)
    return text[:sp[0]] + text[sp[3]:] if sp else text


def _connectivity_only(st):
    """a NETS / SPECIALNETS statement without its wiring (connections and + USE kept)"""
    k = st.find('+')
    head = st[:k] if k >= 0 else st.rstrip().rstrip(';')
    use = re.search(r'\+\s*USE\s+\S+', st)
    return head.rstrip() + (' ' + use.group(0) if use else '') + ' ;'


def def_reductions(def_text, masters):
    """[(key, description, reduced DEF text)]"""
    out = [('no_special_wiring', 'no SPECIALNETS wiring (power / ground straps)',
            _rewrite(def_text, 'SPECIALNETS', _connectivity_only)),
           ('no_pins', 'no I/O pins (PINS section and the nets\' PIN connections)',
            _rewrite(_rewrite(_drop_section(def_text, 'PINS'), 'NETS', lambda s: re.sub(r'\(\s*PIN\s+\S+\s*\)', '', s)),
                     'SPECIALNETS', lambda s: re.sub(r'\(\s*PIN\s+\S+\s*\)', '', s))),
           ('no_net_wiring', 'no NETS wiring (diagnosis only: nothing left to reroute)',
            _rewrite(def_text, 'NETS', _connectivity_only))]
    comp = _section_span(def_text, 'COMPONENTS')
    body = def_text[comp[1]:comp[2]] if comp else ''
    for m in masters:
        insts = set(re.findall(r'^\s*-\s+(\S+)\s+' + re.escape(m) + r'\b', body, re.M))
        if not insts:
            continue
        t = _rewrite(def_text, 'COMPONENTS', lambda s, m=m: None if len(s.split()) > 2 and s.split()[2] == m else s)
        drop = lambda s, insts=insts: re.sub(r'\(\s*(\S+)\s+\S+\s*\)', lambda x: '' if x.group(1) in insts else x.group(0), s)
        t = _rewrite(_rewrite(t, 'NETS', drop), 'SPECIALNETS', drop)
        t = _drop_section(t, 'GROUPS')
        out.append((f'no_cell:{m}', f'no {m} instances ({len(insts)}; their nets kept, not rerouted)', t))
    return out


def _in_macros(text, fn):
    return re.sub(r'^([ \t]*MACRO[ \t]+(\S+)[ \t]*\r?\n)(.*?)(^[ \t]*END[ \t]+\2\b)',
                  lambda m: m.group(1) + fn(m.group(3)) + m.group(4), text, flags=re.M | re.S)


def lef_reductions(lefs):
    """[(key, description, reduced LEF paths)]: copies of the LEFs that contain MACROs"""
    routing = set(C.lef_routing_layers(lefs))
    allayers = set()
    for l in lefs:
        allayers |= set(re.findall(r'^[ \t]*LAYER[ \t]+(\S+)[ \t]*\r?$', Path(l).read_text(errors='replace'), re.M))
    device = allayers - routing
    out = []
    for key, desc, fn in (('cells_metal_only', 'cell pins / OBS without shapes on non-metal layers (poly, diffusion, cuts)',
                           lambda b: P._drop_layer_geometry(b, device)),
                          ('cells_no_obs', 'cells without OBS (blockages inside cells)',
                           lambda b: re.sub(r'^[ \t]*OBS\b.*?^[ \t]*END[ \t]*\r?\n', '', b, flags=re.M | re.S))):
        new = []
        for l in lefs:
            t = Path(l).read_text(errors='replace')
            if re.search(r'^\s*MACRO\s', t, re.M):
                new.append(P._tech_copy(l, _in_macros(t, fn), key))
            else:
                new.append(l)
        out.append((key, desc, new))
    return out


def write_def(def_path, text, key):
    p = Path(def_path)
    tag = re.sub(r'[^A-Za-z0-9_]+', '_', key)
    if len(tag) > 40:   # many cell types in one reduction: a short hash keeps the file name valid
        import hashlib
        tag = tag[:24] + '_' + hashlib.sha256(key.encode()).hexdigest()[:10]
    q = P.CACHE / C.sha256(def_path)[:16] / (p.stem + '.' + tag + '.def')
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text(text)
    return str(q)


def _net_conns(def_path):
    t = Path(def_path).read_text(errors='replace')
    sn = _section_span(t, 'NETS')
    out = {}
    for st in re.split(r'(?<=;)\s*\n', t[sn[1]:sn[2]] if sn else ''):
        f = st.split()
        if len(f) > 1 and f[0] == '-':
            out[f[1].replace(chr(92), '')] = set(re.findall(r'\(\s*(\S+\s+\S+)\s*\)', st.split('+')[0]))
    return out


def nets_touching(def_path, master, other_def=None):
    """NETS names with a connection to an instance of this cell type; with master None and other_def: the nets
    that have fewer connections (cells or I/O pins left out) in other_def than in def_path"""
    if master is None and other_def:
        a, b = _net_conns(def_path), _net_conns(other_def)
        return sorted(n for n, cs in a.items() if n in b and not cs <= b[n])
    t = Path(def_path).read_text(errors='replace')
    sp = _section_span(t, 'COMPONENTS')
    insts = set(re.findall(r'^\s*-\s+(\S+)\s+' + re.escape(master) + r'\b', t[sp[1]:sp[2]], re.M)) if sp else set()
    sn = _section_span(t, 'NETS')
    out = []
    for st in re.split(r'(?<=;)\s*\n', t[sn[1]:sn[2]] if sn else ''):
        f = st.split()
        if len(f) > 1 and f[0] == '-':
            head = st.split('+')[0]
            if any(i in insts for i in re.findall(r'\(\s*(\S+)\s+\S+\s*\)', head)):
                out.append(f[1].replace(chr(92), ''))
    return out


def masters_of(def_path):
    t = Path(def_path).read_text(errors='replace')
    sp = _section_span(t, 'COMPONENTS')
    return sorted(set(re.findall(r'^\s*-\s+\S+\s+(\S+)', t[sp[1]:sp[2]], re.M))) if sp else []


def _masters_named(def_text, errs):
    """cell types of the instances an error names ('... for _099_/A1', 'instance I56')"""
    sp = _section_span(def_text, 'COMPONENTS')
    inst2m = dict(re.findall(r'^\s*-\s+(\S+)\s+(\S+)', def_text[sp[1]:sp[2]], re.M)) if sp else {}
    names = set()
    for e in errs:
        names |= set(re.findall(r'([^\s/(),]+)/[^\s/(),]+', e)) | set(re.findall(r'instance\s+(\S+)', e, re.I))
    return sorted({inst2m[n] for n in names if n in inst2m})


def _norm(e):
    return re.sub(r'(?<![-\w.])\d+(\.\d+)?\b', '#', e.strip())


def repair(case, base_errs, max_steps=6):
    """Cumulative reductions until the router probe passes: a reduction is kept when the errors it was tried against
    are gone (new ones may appear underneath: several causes at once), then the search goes on from there.
    -> (lefs, def, [descriptions applied], remaining error lines)"""
    cur = dict(case)
    errs, applied, used = list(base_errs), [], set()
    for _ in range(max_steps):
        progress = False
        for k, d, e, lf, df in bisect(cur, base_errs=errs, skip_keys=used, lazy=True):
            if not e:
                return lf, df, applied + [d], []
            if not ({_norm(x) for x in errs} & {_norm(x) for x in e}):
                cur = dict(cur, lefs=lf, **{'def': df})
                errs, progress = e, True
                applied.append(d); used.add(k)   # only applied reductions are skipped later; tried ones are retried
                break
        if not progress:
            break
    return cur['lefs'], cur['def'], applied, errs


def bisect(case, include_diagnostic=False, stop_at_first=False, base_errs=(), skip_keys=(), lazy=False):
    """router probe on each reduction of the case -> [(key, description, error lines, lefs, def)];
    least destructive first: cell LEF reductions, special wiring, pins, the cell types the errors name, then one
    cell type at a time (all of them only when there are at most 12 cell types). lazy: a generator."""
    gen = _bisect(case, include_diagnostic, stop_at_first, base_errs, skip_keys)
    return gen if lazy else list(gen)


def _bisect(case, include_diagnostic, stop_at_first, base_errs, skip_keys):
    skip = case.get('skip_tracks', [])
    text = Path(case['def']).read_text(errors='replace')
    cands = [(k, d, lf, case['def']) for k, d, lf in lef_reductions(case['lefs']) if k not in skip_keys]
    named = _masters_named(text, base_errs)
    allm = masters_of(case['def'])
    reds = def_reductions(text, named + [m for m in allm if m not in named] if len(allm) <= 12 else named)
    if len(named) > 1:   # several cell types named by the errors: all of them at once
        t = text
        for m in named:
            t = dict((k, x) for k, _, x in def_reductions(t, [m]))[f'no_cell:{m}']
        reds.insert(2, ('no_cell:' + '+'.join(named), f'no instances of the cell types the errors name ({", ".join(named)})', t))
    for k, d, t in reds:
        if (k == 'no_net_wiring' and not include_diagnostic) or k in skip_keys:
            continue
        cands.append((k, d, case['lefs'], write_def(case['def'], t, k)))
    for k, d, lf, df in cands:
        errs = [e for e in P.router_probe(lf, df, case['layers'], skip) if not e.startswith('(no error line')]
        yield (k, d, errs, lf, df)
        if stop_at_first and not errs:
            return
