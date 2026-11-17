"""DEF cells without a LEF definition: find them, and write a DEF copy without them so the flow can run anyway.

The copy (in out/.cache/, or $LEFDEF_CACHE) drops those components and their pins from the NETS / SPECIALNETS
connection lists; all wiring is kept. Nets that touched a dropped component keep their wires (they still count as
neighbours for coupling and DRC) but must not be rerouted, since their pins on the dropped cells are unknown."""
import hashlib
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def lef_macros(lefs):
    out = set()
    for l in lefs:
        out |= set(re.findall(r'^\s*MACRO\s+(\S+)', Path(l).read_text(errors='replace'), re.M))
    return out


def _section(text, name):
    """(start of the count line, end of the count line, body, count-line match) of a DEF section, or None"""
    m = re.search(r'^([ \t]*' + name + r'[ \t]+)(\d+)([ \t]*;[^\n]*\n)', text, re.M)
    if not m:
        return None
    e = re.search(r'^[ \t]*END[ \t]+' + name + r'\b', text[m.end():], re.M)
    body = text[m.end():m.end() + e.start()] if e else text[m.end():]
    return m, body


def missing_masters(def_path, lefs):
    """sorted cell/macro types used in COMPONENTS that no LEF defines"""
    s = _section(Path(def_path).read_text(errors='replace'), 'COMPONENTS')
    if not s:
        return []
    used = set(re.findall(r'^\s*-\s+\S+\s+(\S+)', s[1], re.M))
    return sorted(used - lef_macros(lefs))


def def_without_masters(def_path, masters):
    """-> (path of the DEF copy, number of components removed, sorted names of the nets that touched them)"""
    text = Path(def_path).read_text(errors='replace')
    removed, touched = set(), set()

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
        if len(f) >= 3 and f[0] == '-' and f[2] in masters:
            removed.add(f[1])
            return None
        return st

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
                        touched.add(net)
                    return ''
                return mm.group(0)
            return re.sub(r'\(\s*(\S+)\s+\S+\s*\)', drop, head) + tail
        return fn

    rewrite('COMPONENTS', comp)
    rewrite('NETS', conns(True))
    rewrite('SPECIALNETS', conns(False))
    out = Path(os.environ.get('LEFDEF_CACHE') or REPO / 'out' / '.cache') / _sha(def_path)[:16] / (Path(def_path).stem + '.known_cells.def')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return str(out), len(removed), sorted(touched)
