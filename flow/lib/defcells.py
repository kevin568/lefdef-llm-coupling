"""Make a DEF readable with the given LEFs, so the flow keeps running when they do not match exactly.

A copy of the DEF (in out/.cache/, or $LEFDEF_CACHE) is written when the DEF uses
  - cells with no LEF MACRO: those components are left out, and their pins are removed from the NETS / SPECIALNETS
    connection lists (wiring is kept); nets that touched them are returned as "fixed" (not rerouted);
  - vias generated from a VIARULE the LEF does not define, or with shapes on layers the LEF does not define (e.g. a
    device-contact via M1_LiAct): the via is rewritten as plain rectangles on the layers the LEF knows (the M1 pad of
    a contact); a via with no known layer left is removed from the wiring;
  - wiring on layers the LEF does not define: those path pieces are removed.
The original DEF is never changed."""
import hashlib
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOK = re.compile(r'\(|\)|;|[^\s();]+')


def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def lef_info(lefs):
    """(macros, layer names of every type, via rule names, fixed via names)"""
    macros, layers, rules, vias = set(), set(), set(), set()
    for l in lefs:
        t = Path(l).read_text(errors='replace')
        macros |= set(re.findall(r'^\s*MACRO\s+(\S+)', t, re.M))
        layers |= set(re.findall(r'^\s*LAYER\s+(\S+)\s*$', t, re.M))
        rules |= set(re.findall(r'^\s*VIARULE\s+(\S+)', t, re.M | re.I))
        vias |= set(re.findall(r'^\s*VIA\s+(\S+)(?:\s+DEFAULT)?\s*$', t, re.M))
    return macros, layers, rules, vias


def lef_macros(lefs):
    return lef_info(lefs)[0]


def _section(text, name):
    m = re.search(r'^([ \t]*' + name + r'[ \t]+)(\d+)([ \t]*;[^\n]*\n)', text, re.M)
    if not m:
        return None
    e = re.search(r'^[ \t]*END[ \t]+' + name + r'\b', text[m.end():], re.M)
    body = text[m.end():m.end() + e.start()] if e else text[m.end():]
    return m, body


def missing_masters(def_path, lefs):
    s = _section(Path(def_path).read_text(errors='replace'), 'COMPONENTS')
    if not s:
        return []
    used = set(re.findall(r'^\s*-\s+\S+\s+(\S+)', s[1], re.M))
    return sorted(used - lef_macros(lefs))


def _generated_rects(st, known):
    """plain rectangles, on known layers, of a via generated from VIARULE parameters (LEF/DEF 5.8 rules)"""
    def nums(key, n):
        m = re.search(r'\+\s*' + key + r'((?:\s+-?\d+){%d})' % n, st)
        return [int(x) for x in m.group(1).split()] if m else None
    lay = re.search(r'\+\s*LAYERS\s+(\S+)\s+(\S+)\s+(\S+)', st)
    if not lay:
        return []
    bot, cut, top = lay.groups()
    cw, ch = nums('CUTSIZE', 2) or [0, 0]
    sx, sy = nums('CUTSPACING', 2) or [0, 0]
    be_x, be_y, te_x, te_y = nums('ENCLOSURE', 4) or [0, 0, 0, 0]
    rows, cols = nums('ROWCOL', 2) or [1, 1]
    ox, oy = nums('ORIGIN', 2) or [0, 0]
    bo_x, bo_y, to_x, to_y = nums('OFFSET', 4) or [0, 0, 0, 0]
    w = cols * cw + (cols - 1) * sx; h = rows * ch + (rows - 1) * sy
    out = []
    for layer, ex, ey, dx, dy in ((bot, be_x, be_y, bo_x, bo_y), (top, te_x, te_y, to_x, to_y)):
        if layer in known:
            out.append((layer, ox + dx - w // 2 - ex, oy + dy - h // 2 - ey, ox + dx + w - w // 2 + ex, oy + dy + h - h // 2 + ey))
    if cut in known:
        x0, y0 = ox - w // 2, oy - h // 2
        for r in range(rows):
            for c in range(cols):
                x = x0 + c * (cw + sx); y = y0 + r * (ch + sy)
                out.append((cut, x, y, x + cw, y + ch))
    return out


def _fix_via(st, known, rules):
    """rewritten VIAS statement, or None to drop the via; st unchanged when it is fine"""
    rule = re.search(r'\+\s*VIARULE\s+(\S+)', st)
    if rule:
        lay = re.search(r'\+\s*LAYERS\s+(\S+)\s+(\S+)\s+(\S+)', st)
        if rule.group(1) in rules and lay and all(x in known for x in lay.groups()):
            return st
        rects = _generated_rects(st, known)
    else:
        shapes = re.findall(r'\+\s*(RECT|POLYGON)\s+(\S+)((?:\s*\(\s*-?\d+\s+-?\d+\s*\))+)', st)
        if all(layer in known for _, layer, _ in shapes):
            return st
        rects = []
        for kind, layer, pts in shapes:
            if layer not in known:
                continue
            xy = [int(v) for v in re.findall(r'-?\d+', pts)]
            xs, ys = xy[0::2], xy[1::2]
            rects.append((layer, min(xs), min(ys), max(xs), max(ys)))
    if len(rects) < 3:   # OpenDB needs at least 3 shapes per via (bottom, cut, top): drop it from the wiring
        return None
    name = st.split()[1]
    lead = st[:len(st) - len(st.lstrip())]
    return (lead + '- ' + name + ' ' + ' '.join('+ RECT %s ( %d %d ) ( %d %d )' % r for r in rects) + ' ;\n')


def _via_tokens(tail):
    """via names used in a NETS / SPECIALNETS wiring text"""
    tk = TOK.findall(tail); out = []; i = 0; routing = False
    while i < len(tk):
        x = tk[i]
        if x == '+':
            kw = tk[i + 1] if i + 1 < len(tk) else ''
            if kw in ('ROUTED', 'FIXED', 'COVER', 'NOSHIELD'):
                routing = True; i += 3; continue
            if routing and kw in ('SHAPE', 'MASK'):
                i += 3; continue
            routing = False; i += 2; continue
        if not routing:
            i += 1; continue
        if x == 'NEW':
            i += 2; continue
        if x == '(':
            i = tk.index(')', i) + 1 if ')' in tk[i:] else len(tk); continue
        if x in ('RECT', 'VIRTUAL'):
            i = tk.index(')', i) + 1 if ')' in tk[i:] else len(tk); continue
        if x in ('MASK', 'STYLE', 'TAPERRULE'):
            i += 2; continue
        if x in ('TAPER', 'N', 'S', 'E', 'W', 'FN', 'FS', 'FE', 'FW', ';') or re.match(r'^-?\d+$', x):
            i += 1; continue
        out.append(x); i += 1
    return out


def _fix_wiring(st, known, dropped_vias):
    """remove path pieces on unknown layers and uses of dropped vias from a NETS / SPECIALNETS statement"""
    k = st.find('+')
    if k < 0:
        return st
    head, tail = st[:k], st[k:]
    # split the wiring into pieces that start at '+ ROUTED|FIXED|COVER|NOSHIELD layer' or 'NEW layer'
    parts = re.split(r'(\+\s*(?:ROUTED|FIXED|COVER|NOSHIELD)\s+|\bNEW\s+)', tail)
    out = [parts[0]]
    first_kw = None
    for i in range(1, len(parts), 2):
        kw, body = parts[i], parts[i + 1]
        # the piece's own wiring vs what follows it ('+ USE ...' attributes or the final ';')
        m = re.search(r'\+\s*(?!SHAPE\b|MASK\b)[A-Z]|;\s*$', body)
        own, rest = (body[:m.start()], body[m.start():]) if m else (body, '')
        layer = own.split()[0] if own.split() else ''
        for v in dropped_vias:
            own = re.sub(r'(?<=\s)' + re.escape(v) + r'(?=\s|$)', '', own)
        points = len(re.findall(r'\(', own))
        has_more = len(own.split()) > 1 + 4 * points   # a via or other element besides the points
        if layer not in known or (points < 2 and not has_more):
            out.append(' ' + rest if rest else '')
            continue
        body = own + rest
        if first_kw is None and not kw.startswith('+'):
            kw = '+ ROUTED '   # the first kept piece must open the wiring
        first_kw = first_kw or kw
        out.append(kw + body)
    return head + ''.join(out)


def sanitize(def_path, lefs):
    """-> (DEF path to use, notes [str], fixed nets [names]); the same path and no notes when nothing is needed"""
    macros, known, rules, lef_vias = lef_info(lefs)
    text = Path(def_path).read_text(errors='replace')
    notes, fixed = [], set()
    removed, dropped_vias, fixed_vias, bad_layers, undefined_vias = set(), set(), [], set(), set()

    def rewrite(name, fn):
        nonlocal text
        s = _section(text, name)
        if not s:
            return
        m, body = s
        parts = re.split(r'(;[ \t]*\r?\n)', body)
        stmts = [parts[k] + (parts[k + 1] if k + 1 < len(parts) else '') for k in range(0, len(parts), 2)]
        kept = [x for x in (fn(st) for st in stmts) if x is not None]
        count = sum(1 for x in kept if x.strip().startswith('-'))
        text = text[:m.start()] + m.group(1) + str(count) + m.group(3) + ''.join(kept) + text[m.end() + len(body):]

    def comp(st):
        f = st.split()
        if len(f) >= 3 and f[0] == '-' and f[2] not in macros:
            removed.add(f[1]); return None
        return st

    def via(st):
        if not st.strip().startswith('-'):
            return st
        new = _fix_via(st, known, rules)
        if new is None:
            dropped_vias.add(st.split()[1])
        elif new is not st:
            fixed_vias.append(st.split()[1])
        return new

    def conns(track):
        def fn(st):
            if not st.strip().startswith('-'):
                return st
            k = st.find('+')
            head, tail = (st[:k], st[k:]) if k >= 0 else (st, '')
            net = st.split()[1]

            def drop(mm):
                if mm.group(1) in removed:
                    if track:
                        fixed.add(net)
                    return ''
                return mm.group(0)
            head = re.sub(r'\(\s*(\S+)\s+\S+\s*\)', drop, head)
            bad = [l for l in re.findall(r'(?:ROUTED|FIXED|COVER|NOSHIELD|NEW)\s+(\S+)', tail) if l not in known]
            undef = {v for v in _via_tokens(tail) if v not in known_vias}
            if not bad and not undef:
                return head + tail   # untouched
            bad_layers.update(bad)   # such nets stay reroutable: their cells' pins are in the LEF
            undefined_vias.update(undef)
            return head + _fix_wiring(tail, known, undef)
        return fn

    other_fixes = []

    def pin(st):   # PINS: shapes / vias on layers or vias the LEF does not define
        if not st.strip().startswith('-'):
            return st
        pt = r'\(\s*-?\d+\s+-?\d+\s*\)'
        opt = r'(?:\s+(?:MASK|SPACING|DESIGNRULEWIDTH)\s+\d+)*'
        def gone(kind, name):
            ok = name in known if kind != 'VIA' else name in known_vias_pre
            if not ok:
                other_fixes.append('pin shape on ' + name)
            return not ok
        st = re.sub(r'\+\s*LAYER\s+(\S+)' + opt + r'\s*' + pt + r'\s*' + pt,
                    lambda m: '' if gone('LAYER', m.group(1)) else m.group(0), st)
        st = re.sub(r'\+\s*POLYGON\s+(\S+)' + opt + r'((?:\s*' + pt + r')+)',
                    lambda m: '' if gone('POLYGON', m.group(1)) else m.group(0), st)
        st = re.sub(r'\+\s*VIA\s+(\S+)(?:\s+MASK\s+\d+)?\s*' + pt,
                    lambda m: '' if gone('VIA', m.group(1)) else m.group(0), st)
        return st

    def layer_stmt(st):   # BLOCKAGES / FILLS: '- LAYER name ...' or '- VIA name ...' statements
        f = st.split()
        if len(f) >= 3 and f[0] == '-' and ((f[1] == 'LAYER' and f[2] not in known) or
                                            (f[1] == 'VIA' and f[2] not in known_vias_pre)):
            other_fixes.append(f[1].lower() + ' ' + f[2]); return None
        return st

    def group(st):   # GROUPS: components that were left out
        if not removed or not st.strip().startswith('-'):
            return st
        k = st.find('+')
        head, tail = (st[:k], st[k:]) if k >= 0 else (st, '')
        f = head.split()
        keep = f[:2] + [x for x in f[2:] if x not in removed]
        return ('    ' + ' '.join(keep) + ' ' + tail) if len(keep) != len(f) else st

    def ndr(st):   # NONDEFAULTRULES: layers / vias / via rules the LEF does not define
        if not st.strip().startswith('-'):
            return st
        st = re.sub(r'\+\s*LAYER\s+(\S+)((?:\s+(?:WIDTH|DIAGWIDTH|SPACING|WIREEXT)\s+\d+)*)',
                    lambda m: m.group(0) if m.group(1) in known else '', st)
        st = re.sub(r'\+\s*VIA\s+(\S+)', lambda m: m.group(0) if m.group(1) in known_vias_pre else '', st)
        st = re.sub(r'\+\s*VIARULE\s+(\S+)', lambda m: m.group(0) if m.group(1) in rules else '', st)
        return st

    rewrite('COMPONENTS', comp)
    rewrite('VIAS', via)
    s0 = _section(text, 'VIAS')
    known_vias_pre = set(lef_vias) | (set(re.findall(r'^\s*-\s+(\S+)', s0[1], re.M)) if s0 else set())
    rewrite('PINS', pin)
    rewrite('BLOCKAGES', layer_stmt)
    rewrite('FILLS', layer_stmt)
    rewrite('GROUPS', group)
    rewrite('NONDEFAULTRULES', ndr)

    def tracks(m):   # TRACKS ... LAYER a b ; with layers the LEF does not define
        layers = m.group(2).split()
        ok = [l for l in layers if l in known]
        if len(ok) == len(layers):
            return m.group(0)
        other_fixes.append('tracks on ' + ' '.join(sorted(set(layers) - set(ok))))
        return (m.group(1) + ' '.join(ok) + m.group(3)) if ok else ''
    text = re.sub(r'^([ \t]*TRACKS\b.*?\bLAYER\s+)([^;\n]+?)(\s*;[^\n]*\n)', tracks, text, flags=re.M)
    s = _section(text, 'VIAS')
    known_vias = set(lef_vias) | (set(re.findall(r'^\s*-\s+(\S+)', s[1], re.M)) if s else set())
    rewrite('NETS', conns(True))
    rewrite('SPECIALNETS', conns(False))
    if removed:
        miss = sorted({m for m in re.findall(r'^\s*-\s+\S+\s+(\S+)', (_section(Path(def_path).read_text(errors='replace'),
                                                                                  'COMPONENTS') or (None, ''))[1], re.M)} - macros)
        notes.append(f'{len(miss)} cell type(s) have no LEF definition ({", ".join(miss[:6])}{" ..." if len(miss) > 6 else ""}): '
                     f'their {len(removed)} instance(s) are left out')
    if fixed_vias or dropped_vias:
        notes.append(f'{len(fixed_vias) + len(dropped_vias)} via(s) use a via rule or layer the LEF does not define (e.g. '
                     f'{(fixed_vias + sorted(dropped_vias))[0]}): kept as rectangles on the LEF layers'
                     + (f', {len(dropped_vias)} with no LEF layer removed' if dropped_vias else ''))
    if undefined_vias - dropped_vias:
        u = sorted(undefined_vias - dropped_vias)
        notes.append(f'{len(u)} via name(s) used in the wiring are defined in no usable LEF or DEF ({", ".join(u[:6])}'
                     f'{" ..." if len(u) > 6 else ""}): left out of the wiring')
    if other_fixes:
        notes.append(f'{len(other_fixes)} other item(s) on layers / vias the LEF does not define are left out '
                     f'(e.g. {", ".join(sorted(set(other_fixes))[:4])})')
    if bad_layers:
        notes.append(f'wiring on layer(s) the LEF does not define ({", ".join(sorted(bad_layers)[:4])}) is left out')
    if not notes:
        return str(def_path), [], []
    out = Path(os.environ.get('LEFDEF_CACHE') or REPO / 'out' / '.cache') / _sha(def_path)[:16] / (Path(def_path).stem + '.for_lef.def')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    if fixed:
        notes.append(f'the {len(fixed)} net(s) touching the left-out cells keep their wiring (counted as neighbours) but are not rerouted')
    return str(out), notes, sorted(fixed)
