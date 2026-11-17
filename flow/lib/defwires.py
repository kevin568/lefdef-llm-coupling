"""Wire rectangles and via counts of every net, read from the DEF NETS section (pure Python).

Used by export_geo.py only when the OpenROAD build has no odb.dbWirePathItr (older releases, e.g. the conda
litex-hub build of 2024). Rectangles follow OpenDB's wire path shapes: a segment of the layer width with width/2
end extensions (or the point's extension value); patch RECTs and VIRTUAL moves are not wires. Checked identical to
dbWirePathItr shapes on 6 layouts (SKY130, Nangate45, IHP SG13G2)."""
import re

TOK = re.compile(r'\(|\)|;|[^\s();]+')
ORIENTS = ('N', 'S', 'E', 'W', 'FN', 'FS', 'FE', 'FW')


def _nets_section(text):
    m = re.search(r'^\s*NETS\s+\d+\s*;', text, re.M)
    if not m:
        return ''
    e = re.search(r'^\s*END\s+NETS\b', text[m.end():], re.M)
    return text[m.end():m.end() + e.start()] if e else text[m.end():]


def net_wires(def_path, width_of, via_layers):
    """{net: (shapes [(layer, (x1, y1, x2, y2))], vias)}. width_of(layer) -> DBU width or 0 (unknown layer);
    via_layers(via) -> set of routing layer names of that via."""
    with open(def_path, errors='replace') as f:
        text = f.read()
    out = {}
    for st in re.split(r';\s*\n', _nets_section(text)):
        st = st.strip()
        if not st.startswith('-'):
            continue
        tk = TOK.findall(st)
        name = tk[1]; i = 2
        while i < len(tk) and tk[i] == '(':
            i = tk.index(')', i) + 1
        shapes = []; vias = 0; layer = None; prev = None; routing = False; width = 0
        while i < len(tk):
            x = tk[i]
            if x == '+':
                kw = tk[i + 1] if i + 1 < len(tk) else ''
                if kw in ('ROUTED', 'FIXED', 'COVER', 'NOSHIELD'):
                    routing = True; layer = tk[i + 2]; width = width_of(layer); prev = None; i += 3; continue
                if routing and kw in ('SHAPE', 'MASK'):
                    i += 3; continue
                routing = False; i += 2; continue
            if not routing:
                i += 1; continue
            if x == 'NEW':
                layer = tk[i + 1]; width = width_of(layer); prev = None; i += 2; continue
            if x == '(':
                j = tk.index(')', i); v = tk[i + 1:j]
                px = prev[0] if v[0] == '*' and prev else int(float(v[0]))
                py = prev[1] if len(v) > 1 and v[1] == '*' and prev else int(float(v[1]))
                ext = int(float(v[2])) if len(v) > 2 and v[2] != '*' else None
                cur = (px, py, ext)
                if prev and (prev[0], prev[1]) != (px, py) and width:
                    hw = width // 2
                    e1 = prev[2] if prev[2] is not None else hw
                    e2 = ext if ext is not None else hw
                    if prev[1] == py:
                        lo, hi = (prev, cur) if prev[0] < px else (cur, prev)
                        elo = e1 if lo is prev else e2; ehi = e2 if hi is cur else e1
                        shapes.append((layer, (lo[0] - elo, py - hw, hi[0] + ehi, py + hw)))
                    elif prev[0] == px:
                        lo, hi = (prev, cur) if prev[1] < py else (cur, prev)
                        elo = e1 if lo is prev else e2; ehi = e2 if hi is cur else e1
                        shapes.append((layer, (px - hw, lo[1] - elo, px + hw, hi[1] + ehi)))
                prev = cur; i = j + 1; continue
            if x == 'RECT':
                i = tk.index(')', i) + 1; continue
            if x == 'VIRTUAL':
                j = tk.index(')', i); v = tk[i + 2:j]
                prev = (int(float(v[0])), int(float(v[1])), None); i = j + 1; continue
            if x in ('MASK', 'STYLE', 'TAPERRULE'):
                i += 2; continue
            if x == 'TAPER' or x in ORIENTS or re.match(r'^-?\d+$', x):
                i += 1; continue
            vias += 1   # a via: the path continues on its other routing layer
            other = [l for l in via_layers(x) if l != layer]
            if other:
                layer = other[0]; width = width_of(layer)
            i += 1
        out[name] = (shapes, vias)
    return out
