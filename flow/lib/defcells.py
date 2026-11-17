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


def macro_pins(lefs):
    """{macro: set of its PIN names}"""
    out = {}
    for l in lefs:
        t = Path(l).read_text(errors='replace')
        for m in re.finditer(r'^\s*MACRO\s+(\S+)\s*$', t, re.M):
            e = re.search(r'^\s*END\s+' + re.escape(m.group(1)) + r'\s*$', t[m.end():], re.M)
            body = t[m.end():m.end() + e.start()] if e else t[m.end():]
            out.setdefault(m.group(1), set()).update(re.findall(r'^\s*PIN\s+(\S+)', body, re.M))
    return out


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
        if not re.match(r'^[^\s+();]+$', layer) or re.match(r'^-?\d+$', layer):   # no layer after the keyword: keep
            out.append(kw + body)
            first_kw = first_kw or kw
            continue
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


NAMED = ('COMPONENTS', 'VIAS', 'PINS', 'GROUPS', 'NONDEFAULTRULES', 'NETS', 'SPECIALNETS', 'REGIONS', 'SCANCHAINS',
         'STYLES', 'PINPROPERTIES')   # sections whose statements start '- name'
SECTIONS = NAMED + ('BLOCKAGES', 'FILLS')


def stmt_key(section, st):
    """identity of a DEF section statement: its name, or (BLOCKAGES / FILLS, unnamed) its text"""
    return (section, st.split()[1] if section in NAMED else ' '.join(st.split()))


def statement_at(def_path, line_no):
    """what to leave out so OpenROAD gets past an error reported at this DEF line: ('stmt', key) or ('line', text)"""
    lines = Path(def_path).read_text(errors='replace').splitlines()
    if not 1 <= line_no <= len(lines):
        return None
    i = line_no - 1
    # an error at a section header or END line (e.g. a section left empty): the whole section, never the line alone
    se = re.match(r'^\s*(?:END\s+)?([A-Z]+)\b(?:\s+\d+\s*;)?\s*$', lines[i])
    if se and se.group(1) not in ('DESIGN', 'VERSION') and re.search(r'^\s*END\s+' + se.group(1) + r'\b', '\n'.join(lines), re.M):
        return ('section', se.group(1))
    head = re.compile(r'^\s*(' + '|'.join(SECTIONS) + r')\s+\d+\s*;')
    for j in range(i, -1, -1):   # the enclosing section, if any
        if j < i and re.match(r'^\s*END\s+\S+', lines[j]):
            break
        h = head.match(lines[j])
        if not h:
            continue
        if j == i:
            return None
        k = i   # the statement start: the nearest '- ' line at or above, with no statement end in between
        while k > j and not lines[k].lstrip().startswith('-'):
            k -= 1
            if lines[k].rstrip().endswith(';') and not lines[k].lstrip().startswith('-'):
                k = j
                break
        if k == j:
            break
        end = k
        while end < len(lines) - 1 and not lines[end].rstrip().endswith(';'):
            end += 1
        return ('stmt', stmt_key(h.group(1), ' '.join(lines[k:end + 1])))
    return ('line', lines[i].strip()) if lines[i].strip() else None


def sanitize(def_path, lefs, drop=None, top_layer=None, above=()):
    """-> (DEF path to use, notes [str], fixed nets [names]); the same path and no notes when nothing is needed.
    drop: statement keys / line texts to leave out (what OpenROAD still could not read, see statement_at)."""
    macros, known, rules, lef_vias = lef_info(lefs)
    text = Path(def_path).read_text(errors='replace')
    notes, fixed = [], set()
    removed, dropped_vias, fixed_vias, bad_layers, undefined_vias = set(), set(), [], set(), set()
    drop = drop or set()
    left_out, dups = [], []
    # multi-patterning colours: OpenROAD 2.0 (conda) writes '( x y )MASK 1' back out and cannot read its own DEF;
    # the flow does not use colours, so the copy goes without them
    n_mask = len(re.findall(r'\bMASK\s+[0-9A-Fa-f]+\b|\bMASKSHIFT\s+\S+', text))
    if n_mask:
        text = re.sub(r'^[ \t]*COMPONENTMASKSHIFT\b[^;]*;[^\n]*\n', '', text, flags=re.M)
        text = re.sub(r'(?:\+\s*)?\bMASKSHIFT\s+[^\s;]+', ' ', text)
        text = re.sub(r'(?:\+\s*)?\bMASK\s+[0-9A-Fa-f]+\b', ' ', text)
    drop_lines = {x[1] for x in drop if x[0] == 'line'}
    drop_names = {x[1] for x in drop if x[0] == 'name'} - known - lef_vias   # never a layer or a LEF via name
    drop_names = {n for n in drop_names if not re.match(r'^-?[\d.]+$', n)}
    emptied = []
    for sec in sorted({x[1] for x in drop if x[0] == 'section'} - {'DESIGN'}):   # whole sections OpenROAD cannot read
        text, k = re.subn(r'^[ \t]*' + sec + r'\b[^\n]*\n.*?^[ \t]*END[ \t]+' + sec + r'\b[^\n]*\n?', '', text, count=1,
                          flags=re.M | re.S)
        if k:
            emptied.append(sec)
    drop_conns = {(x[1], x[2]) for x in drop if x[0] == 'conn'}
    mpins, no_pin, to_routed, pins_top = macro_pins(lefs), [], [], []
    above = set(above)
    top_side = 0
    if top_layer and above:
        from .lefpins import _layer_size
        w, p, _ = _layer_size(lefs, top_layer)
        u = re.search(r'^\s*UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', text, re.M)
        top_side = int(round(max(2 * w, p) * (int(u.group(1)) if u else 1000)))
        top_side += top_side % 2
    s_comp = _section(text, 'COMPONENTS')
    master = dict(re.findall(r'^\s*-\s+(\S+)\s+(\S+)', s_comp[1], re.M)) if s_comp else {}
    if drop_lines:
        text = ''.join(l for l in text.splitlines(True) if l.strip() not in drop_lines)
        left_out += sorted(drop_lines)

    def rewrite(name, fn):
        nonlocal text
        s = _section(text, name)
        if not s:
            return
        m, body = s
        parts = re.split(r'(;[ \t]*\r?\n)', body)
        stmts = [parts[k] + (parts[k + 1] if k + 1 < len(parts) else '') for k in range(0, len(parts), 2)]
        seen = set()

        def gate(st):   # leave out statements OpenROAD could not read, and repeated definitions of a name (first kept)
            if st.strip().startswith('-') and len(st.split()) > 1:
                for bad in drop_names:   # a name an error reported: drop the '+ KEYWORD name' item, else the statement
                    if bad in st.split() or (bad + ';') in st.split():
                        st2 = re.sub(r'\+\s*[A-Z]+\s+' + re.escape(bad) + r'(?=[\s;])', '', st)
                        if st2 != st and bad not in st2.replace(';', ' ').split():
                            left_out.append(f'{name} {st.split()[1][:40]}: + ... {bad}')
                            st = st2
                        else:
                            left_out.append(f'{name} {st.split()[1][:60]}')
                            if name == 'VIAS':
                                dropped_vias.add(st.split()[1])
                            return None
                key = stmt_key(name, st)
                if ('stmt', key) in drop:
                    left_out.append(f'{name} {key[1][:60]}')
                    if name == 'VIAS':
                        dropped_vias.add(key[1])
                    return None
                if name in NAMED and key in seen:
                    dups.append(f'{name} {key[1][:60]}')
                    return None
                seen.add(key)
            return fn(st)
        kept = [x for x in (gate(st) for st in stmts) if x is not None]
        count = sum(1 for x in kept if x.strip().startswith('-'))
        if count == 0 and any(st.strip().startswith('-') for st in stmts) and name not in ('NETS', 'COMPONENTS'):
            emptied.append(name)   # every statement left out: the whole section goes (an empty one is a parse error)
            e = re.search(r'^[ \t]*END[ \t]+' + name + r'\b[^\n]*\n?', text[m.end() + len(body):], re.M)
            text = text[:m.start()] + text[m.end() + len(body) + (e.end() if e else 0):]
            return
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
                inst, pin = mm.group(1), mm.group(2)
                if inst in removed:
                    if track:
                        fixed.add(net)
                    return ''
                pins = mpins.get(master.get(inst))
                if (inst, pin) in drop_conns or (pins is not None and pin not in pins):
                    no_pin.append((inst, pin))   # a pin the cell's LEF does not have (e.g. bulk B): connection left out
                    return ''
                return mm.group(0)
            head = re.sub(r'\(\s*(\S+)\s+(\S+)\s*\)', drop, head)
            if track and not re.search(r'\bUSE\s+(POWER|GROUND)\b', tail):   # FIXED / COVER signal wiring: reroutable
                tail, k = re.subn(r'\+\s*(?:FIXED|COVER)(?=\s)', '+ ROUTED', tail)
                if k:
                    to_routed.append(net)
            bad = [l for l in re.findall(r'\b(?:ROUTED|FIXED|COVER|NOSHIELD|NEW)\s+([^\s+();]+)', tail)
                   if l not in known and not re.match(r'^-?\d+$', l)]
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
        if top_layer and above:   # I/O pin only above the top signal layer: the router cannot reach it from below
            shape = r'\+\s*LAYER\s+(\S+)' + opt + r'\s*(' + pt + r')\s*(' + pt + r')'
            lays = re.findall(r'\+\s*LAYER\s+(\S+)', st)
            if lays and all(l in above for l in lays):
                def copy(m):   # at least one pitch of the top layer wide, so the detailed router finds an access point
                    (x1, y1), (x2, y2) = [tuple(int(v) for v in re.findall(r'-?\d+', p)) for p in (m.group(2), m.group(3))]
                    x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    hw, hh = max((x2 - x1) // 2, top_side // 2), max((y2 - y1) // 2, top_side // 2)
                    return f'{m.group(0)} + LAYER {top_layer} ( {cx - hw} {cy - hh} ) ( {cx + hw} {cy + hh} )'
                st = re.sub(shape, copy, st)
                pins_top.append(st.split()[1])
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
    for sec in ('REGIONS', 'SCANCHAINS', 'STYLES', 'PINPROPERTIES'):
        rewrite(sec, lambda st: st)

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
    if pins_top:
        notes.append(f'{len(pins_top)} I/O pin(s) only above {top_layer} (e.g. {", ".join(pins_top[:3])}): '
                     f'a copy of the shape on {top_layer} is added so the router can reach them')
    if to_routed:
        notes.append(f'{len(to_routed)} signal net(s) have FIXED / COVER wiring: written as ROUTED in the copy so they can be rerouted')
    if n_mask:
        notes.append(f'{n_mask} multi-patterning MASK colour mark(s) are left out of the copy (not used by the flow; '
                     'OpenROAD 2.0 cannot read them back after writing them)')
    if no_pin:
        notes.append(f'{len(no_pin)} connection(s) to pins the cell LEF does not define are left out (e.g. '
                     f'{" ".join("%s/%s" % x for x in no_pin[:3])}; usually the transistor bulk pin)')
    if emptied:
        notes.append(f'section(s) {", ".join(sorted(set(emptied)))} left out as a whole (nothing in them OpenROAD can use)')
    if dups:
        notes.append(f'{len(dups)} repeated definition(s) of a name are left out, the first kept (e.g. {dups[0]})')
    if left_out:
        notes.append(f'{len(left_out)} item(s) OpenROAD could not read are left out (e.g. {"; ".join(left_out[:3])})')
    if not notes:
        return str(def_path), [], []
    out = Path(os.environ.get('LEFDEF_CACHE') or REPO / 'out' / '.cache') / _sha(def_path)[:16] / (Path(def_path).stem + '.for_lef.def')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    if fixed:
        notes.append(f'the {len(fixed)} net(s) touching the left-out cells keep their wiring (counted as neighbours) but are not rerouted')
    return str(out), notes, sorted(fixed)
