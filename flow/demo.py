#!/usr/bin/env python3
"""Demo: run the whole flow with a clean screen and draw the reroutes.

    python3 flow/demo.py A.def TECH.lef CELLS.lef [run.py options]

Runs flow/run.py with the same inputs (finished stages are reused, so a second call only redraws), keeps its full
output in out/<case>/demo/run.log, prints one line per stage and the before/after result, and writes
out/<case>/demo/index.html: the layout before / after with the rerouted nets marked, and per rerouted net its old and
new route side by side with the coupling (same-layer parallel runs to other nets) shaded.

Options of its own: --no-run (only draw from an existing output folder), --draw N (draw at most N nets, default 8),
--rankers (also print the LLM / heuristic / random comparison). Any other option is passed on to run.py.
Two final layouts are shown side by side: DRC not relaxed (--drc-relax none: no reroute that adds a DRC violation)
and slightly relaxed (abstract: clashes with other cells' LEF pin / OBS abstracts tolerated); --levels changes the list,
--drc-relax X shows only X. The candidates and LLM scores are shared, only the final layout is built per level."""
import argparse
import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

FLOW = Path(__file__).resolve().parent
REPO = FLOW.parent
STEPS = [('s0_check', 'check inputs and tools'), ('s1_import', 'read the layout'), ('s2_candidates', 'build candidate trees'),
         ('s3_score', 'rank the trees (LLM)'), ('s4_route', 'reroute the candidates'), ('s5_extract', 'measure coupling'),
         ('s6_apply', 'build the final layout'), ('s7_report', 'write the report')]
LABEL = dict(STEPS)

# figure colours
C_OTHER = '#a9b4bd'
C_NET = '#00796b'
C_NEIGH = '#d9822b'
C_BAND = '#d33b2c'
C_PIN = '#1b2b34'
C_TEXT = '#1d3540'
C_SUB = '#56707a'
NET_COLORS = ['#00796b', '#3f6fd8', '#c2185b', '#7b52c4', '#d9822b', '#2e8b3d', '#b8860b', '#0097a7']


def clean(n):
    return n.replace('\\', '')


def fmt_pct(a, b):
    return '%+.1f%%' % (100. * (a - b) / b) if b else '-'


# ----------------------------------------------------------------------------------------------- run the flow

def run_flow(argv, log_path):
    """run.py with these arguments; one screen line per stage; everything into log_path -> exit code"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tty = sys.stdout.isatty()
    cmd = [sys.executable, '-u', str(FLOW / 'run.py')] + argv
    t0 = time.time(); cur = None; tail = []; done = set()
    with open(str(log_path), 'w') as log:
        log.write('$ ' + ' '.join(cmd) + '\n'); log.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1,
                             cwd=str(REPO))
        for line in p.stdout:
            log.write(line); log.flush()
            s = line.rstrip('\n'); tail = (tail + [s])[-40:]
            m = re.match(r'\[(run |skip| ok )\] (s\d_\w+)', s)
            if 'Installing the CPU build' in s:
                print('  preparing the LLM (first run only: download and install, several minutes) ...', flush=True)
            if m:
                kind, st = m.group(1), m.group(2)
                n = [x for x, _ in STEPS].index(st) + 1 if st in LABEL else 0
                head = '  [%d/%d] %-26s' % (n, len(STEPS), LABEL.get(st, st))
                if kind == 'run ':
                    cur = (st, time.time())
                    if tty:
                        sys.stdout.write(head + ' ...'); sys.stdout.flush()
                elif st not in done:
                    done.add(st)
                    if kind == 'skip':
                        msg = 'done (reused)'
                    else:
                        info = {}
                        try:
                            info = json.loads(s.split('  ', 1)[1]) if '  ' in s else {}
                        except ValueError:
                            pass
                        msg = 'done  %5.0f s%s' % (time.time() - cur[1] if cur else 0, stage_note(st, info))
                    print(('\r' if tty else '') + head + ' ' + msg + ' ' * 10, flush=True)
                continue
            mm = re.search(r'LLM scored (\d+)/(\d+) prompts.*?about ([\d.]+) min left', s)
            if mm and tty and cur:
                sys.stdout.write('\r  [4/%d] %-26s %s/%s trees, about %s min left   ' % (len(STEPS), LABEL['s3_score'], mm.group(1),
                                                                                      mm.group(2), mm.group(3)))
                sys.stdout.flush()
        p.wait()
    if p.returncode:
        print('\n  the flow stopped (exit %d). Last lines:' % p.returncode)
        keep = [x for x in tail if x.strip() and 'warn' not in x.lower()][-12:]
        for x in keep:
            print('    ' + x[:200])
        print('  full output: %s' % log_path)
    else:
        print('  flow finished in %.0f s (full output: %s)' % (time.time() - t0, log_path))
    return p.returncode


def stage_note(st, i):
    if st == 's1_import' and i.get('signal_nets'):
        return '   %s signal nets' % i['signal_nets']
    if st == 's2_candidates' and i.get('candidates'):
        return '   %s trees for %s nets' % (i['candidates'], i.get('nets', '?'))
    if st == 's3_score' and i.get('scored'):
        return '   %s trees ranked' % i['scored']
    if st == 's4_route' and 'routed' in i:
        return '   %s reroutes' % i['routed']
    if st == 's5_extract' and 'valid' in i:
        return '   %s of %s usable' % (i['valid'], i.get('measured', '?'))
    if st == 's6_apply' and 'accepted' in i:
        return '   %s nets rerouted' % i['accepted']
    return ''


# ----------------------------------------------------------------------------------------------- geometry

def runs(shapes, pitch, dbu):
    """[(layer, 'H'|'V', track_lo, track_hi, a, b)]: merged straight wire runs (the proxy rule: >= 1 um, aspect >= 3)"""
    groups = {}
    for s in shapes:
        layer, r = s['layer'], s['rect']
        if layer not in pitch:
            continue
        w, h = r[2] - r[0], r[3] - r[1]
        if w >= h:
            if w < dbu or w < 3.0 * max(h, 1):
                continue
            groups.setdefault((layer, 'H', r[1], r[3]), []).append((r[0], r[2]))
        else:
            if h < dbu or h < 3.0 * max(w, 1):
                continue
            groups.setdefault((layer, 'V', r[0], r[2]), []).append((r[1], r[3]))
    out = []
    for (layer, o, lo, hi), iv in groups.items():
        iv.sort(); cur = list(iv[0])
        for a, b in iv[1:]:
            if a <= cur[1]:
                cur[1] = max(cur[1], b)
            else:
                out.append((layer, o, lo, hi, cur[0], cur[1])); cur = [a, b]
        out.append((layer, o, lo, hi, cur[0], cur[1]))
    return out


def coupling_bands(geo, net):
    """the parallel runs of `net` to other nets: [(rect between the two wires, other net, um)] and {other net: um}"""
    dbu = geo['dbu']
    pitch = {k: l['pitch'] / float(dbu) for k, l in geo['layers'].items() if l.get('pitch')}
    mine = runs(geo['nets'][net]['shapes'], pitch, dbu)
    if not mine:
        return [], {}
    keys = set((r[0], r[1]) for r in mine)
    others = []
    for n, v in geo['nets'].items():
        if n == net:
            continue
        for r in runs(v['shapes'], pitch, dbu):
            if (r[0], r[1]) in keys:
                others.append((r, n))
    bands = []; per = {}
    for layer, o, lo1, hi1, a1, b1 in mine:
        g = 2.0 * pitch[layer] * dbu
        for (l2, o2, lo2, hi2, a2, b2), n in others:
            if l2 != layer or o2 != o:
                continue
            gap = max(lo2 - hi1, lo1 - hi2)
            if gap <= 0 or gap > g:
                continue
            a, b = max(a1, a2), min(b1, b2)
            if b <= a:
                continue
            t0, t1 = (hi1, lo2) if lo2 >= hi1 else (hi2, lo1)
            rect = [a, t0, b, t1] if o == 'H' else [t0, a, t1, b]
            bands.append((rect, n, (b - a) / float(dbu))); per[n] = per.get(n, 0.) + (b - a) / float(dbu)
    return bands, per


class Frame:
    """maps a design window (dbu) into a pixel box, y up"""

    def __init__(self, win, x, y, w, h):
        x0, y0, x1, y1 = win
        s = min(w / float(max(x1 - x0, 1)), h / float(max(y1 - y0, 1)))
        self.s = s; self.x0 = x0; self.y1 = y1
        self.ox = x + (w - (x1 - x0) * s) / 2.; self.oy = y + (h - (y1 - y0) * s) / 2.
        self.win = win

    def rect(self, r, minpx=0.6):
        x0, y0, x1, y1 = self.win
        a, b, c, d = max(r[0], x0), max(r[1], y0), min(r[2], x1), min(r[3], y1)
        if c < a or d < b:
            return None
        px, py = self.ox + (a - self.x0) * self.s, self.oy + (self.y1 - d) * self.s
        return '%.2f' % px, '%.2f' % py, '%.2f' % max((c - a) * self.s, minpx), '%.2f' % max((d - b) * self.s, minpx)


def svg_rects(F, shapes, color, op, title, minpx=0.6, extra=''):
    out = []
    for s in shapes:
        r = s['rect'] if isinstance(s, dict) else s
        q = F.rect(r, minpx)
        if q:
            t = title(s) if callable(title) else title
            out.append('<rect x="%s" y="%s" width="%s" height="%s" fill="%s" fill-opacity="%s"%s><title>%s</title></rect>'
                       % (q + (color, op, extra, html.escape(t))))
    return out


def level(geo, s):
    return geo['layers'].get(s['layer'], {}).get('level', 0)


def bbox(shapes, pins=()):
    xs = [s['rect'][0] for s in shapes] + [p['xy'][0] for p in pins]
    ys = [s['rect'][1] for s in shapes] + [p['xy'][1] for p in pins]
    xe = [s['rect'][2] for s in shapes] + [p['xy'][0] for p in pins]
    ye = [s['rect'][3] for s in shapes] + [p['xy'][1] for p in pins]
    return [min(xs), min(ys), max(xe), max(ye)] if xs else None


def panel(geo, net, F, label, stats):
    """one side of a net figure: other wiring grey, the net's coupled neighbours orange, the net, coupling bands red"""
    bands, per = coupling_bands(geo, net)
    L = []
    for n, v in geo['nets'].items():
        if n == net:
            continue
        col, op = (C_NEIGH, '.6') if n in per else (C_OTHER, '.45')
        sh = sorted(v['shapes'], key=lambda s: level(geo, s))
        L += svg_rects(F, sh, col, op, lambda s, n=n: '%s  %s' % (clean(n), s['layer']))
    L += svg_rects(F, [b[0] for b in bands], C_BAND, '.55',
                   lambda r: 'coupling', minpx=1.2)
    me = sorted(geo['nets'][net]['shapes'], key=lambda s: level(geo, s))
    L += svg_rects(F, me, C_NET, '.95', lambda s: '%s  %s (rerouted net)' % (clean(net), s['layer']))
    for p in geo['nets'][net].get('pins', []):
        x, y = p['xy']; q = F.rect([x, y, x, y])
        if q:
            L.append('<rect x="%.2f" y="%.2f" width="5" height="5" fill="%s"><title>pin %s</title></rect>'
                     % (float(q[0]) - 2.5, float(q[1]) - 2.5, C_PIN, html.escape(p['name'])))
    top = per and sorted(per.items(), key=lambda kv: -kv[1])[:3]
    near = ', '.join('%s %.1f' % (clean(n), u) for n, u in top) if top else 'none'
    return L, near


def net_figure(before, after, net, rec, idx, total):
    W, H = 1400, 720
    sb, sa = before['nets'][net], after['nets'][net]
    bb = bbox(sb['shapes'] + sa['shapes'], sb.get('pins', []))
    m = max(int(0.18 * max(bb[2] - bb[0], bb[3] - bb[1])), 2 * before['dbu'])
    win = [bb[0] - m, bb[1] - m, bb[2] + m, bb[3] + m]
    Fb = Frame(win, 30, 130, 650, 540); Fa = Frame(win, 720, 130, 650, 540)
    Lb, nb = panel(before, net, Fb, 'before', None)
    La, na = panel(after, net, Fa, 'after', None)
    cb, ca = rec['cc_before_fF'], rec['cc_after_fF']
    wb, wa = rec.get('wl_before_um'), rec.get('wl_after_um')
    tree = rec['candidate_id'].split(':')[-1]
    unit = rec.get('_unit', 'um')
    pol = rec.get('_policy', 'llm')
    who = {'llm': 'LLM', 'hybrid': 'LLM + heuristic', 'heuristic': 'heuristic', 'random': 'random'}.get(pol, pol)
    head = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="Segoe UI, Arial, sans-serif">' % (W, H),
        '<rect width="%d" height="%d" fill="#fbfcfb"/>' % (W, H),
        '<text x="30" y="42" font-size="26" font-weight="700" fill="%s">%s: coupling %.2f &#8594; %.2f %s (%s)</text>'
        % (C_TEXT, html.escape(clean(net)), cb, ca, unit, fmt_pct(ca, cb)),
        '<text x="30" y="92" font-size="19" font-weight="600" fill="%s">Before: original route</text>' % C_TEXT,
        '<text x="30" y="116" font-size="15" fill="%s">coupling %.2f %s%s | main neighbours: %s</text>'
        % (C_SUB, cb, unit, (' | wire %.1f um' % wb) if wb is not None else '', html.escape(nb)),
        '<text x="720" y="92" font-size="19" font-weight="600" fill="%s">After: %s-selected tree (%s), rerouted</text>' % (C_TEXT, who, html.escape(tree)),
        '<text x="720" y="116" font-size="15" fill="%s">coupling %.2f %s%s | main neighbours: %s</text>'
        % (C_SUB, ca, unit, (' | wire %.1f um' % wa) if wa is not None else '', html.escape(na)),
        '<rect x="30" y="130" width="650" height="540" fill="none" stroke="#d5dfdc"/>',
        '<rect x="720" y="130" width="650" height="540" fill="none" stroke="#d5dfdc"/>',
    ]
    leg = legend(30, 700, [(C_NET, 'rerouted net'), (C_NEIGH, 'coupled neighbour'), (C_BAND, 'coupling (parallel run)'),
                           (C_OTHER, 'other wiring'), (C_PIN, 'pin')])
    return '\n'.join(head + ['<g>'] + Lb + ['</g><g>'] + La + ['</g>'] + leg + ['</svg>'])


def legend(x, y, items):
    L = []
    for c, t in items:
        L.append('<rect x="%d" y="%d" width="14" height="14" fill="%s"/>' % (x, y - 12, c))
        L.append('<text x="%d" y="%d" font-size="15" fill="%s">%s</text>' % (x + 20, y, C_SUB, html.escape(t)))
        x += 30 + 8 * len(t)
    return L


def overview_figure(before, after, nets):
    W, H = 1400, 760
    # the area around the rerouted nets (the die and the I/O wiring are often much larger), within the wired area
    full = bbox([s for g in (before, after) for v in g['nets'].values() for s in v['shapes']]) or before.get('die')
    die = bbox([s for g in (before, after) for n in nets if n in g['nets'] for s in g['nets'][n]['shapes']]) or full
    m = int(0.3 * max(die[2] - die[0], die[3] - die[1]))
    die = [max(die[0] - m, full[0]), max(die[1] - m, full[1]), min(die[2] + m, full[2]), min(die[3] + m, full[3])]
    Fb = Frame(die, 30, 90, 650, 610); Fa = Frame(die, 720, 90, 650, 610)
    col = {n: NET_COLORS[i % len(NET_COLORS)] for i, n in enumerate(nets)}
    L = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="Segoe UI, Arial, sans-serif">' % (W, H),
         '<rect width="%d" height="%d" fill="#fbfcfb"/>' % (W, H),
         '<text x="30" y="50" font-size="22" font-weight="600" fill="%s">Before</text>' % C_TEXT,
         '<text x="720" y="50" font-size="22" font-weight="600" fill="%s">After (%d rerouted net%s)</text>'
         % (C_TEXT, len(nets), '' if len(nets) == 1 else 's'),
         '<rect x="30" y="90" width="650" height="610" fill="none" stroke="#d5dfdc"/>',
         '<rect x="720" y="90" width="650" height="610" fill="none" stroke="#d5dfdc"/>']
    for geo, F in ((before, Fb), (after, Fa)):
        L.append('<g>')
        for n, v in geo['nets'].items():
            if n in col:
                continue
            L += svg_rects(F, sorted(v['shapes'], key=lambda s: level(geo, s)), C_OTHER, '.4',
                           lambda s, n=n: '%s  %s' % (clean(n), s['layer']), minpx=0.4)
        for n in nets:
            if n in geo['nets']:
                L += svg_rects(F, geo['nets'][n]['shapes'], col[n], '.95', lambda s, n=n: '%s  %s' % (clean(n), s['layer']), minpx=2.2)
        L.append('</g>')
    L += legend(30, 735, [(col[n], clean(n)) for n in nets[:8]] + [(C_OTHER, 'other wiring')])
    L.append('</svg>')
    return '\n'.join(L)


# ----------------------------------------------------------------------------------------------- page

PAGE_CSS = """
:root{--bg:#eef3f1;--card:#fbfcfb;--ink:#1d3540;--sub:#56707a;--acc:#00796b;--line:#d5dfdc}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:'Segoe UI',Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 60px}h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}
.lead{color:var(--sub);margin:0 0 22px;font-size:16px}.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:0 0 22px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;min-width:200px;flex:1}
.kpi b{display:block;font-size:30px;color:var(--acc);letter-spacing:-.03em}.kpi span{color:var(--sub);font-size:14px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin:0 0 20px}
h2{font-size:20px;margin:0 0 10px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:15px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--sub);font-weight:600;font-size:13px}td.g{color:var(--acc);font-weight:700}
figure{margin:0;cursor:zoom-in}figure svg{width:100%;height:auto;display:block}figcaption{color:var(--sub);font-size:14px;margin-top:6px}
#lb{position:fixed;inset:0;background:rgba(10,30,30,.82);display:none;align-items:center;justify-content:center;z-index:9;cursor:zoom-out}
#lb.on{display:flex}#lb div{width:96vw;max-height:94vh;overflow:auto;background:#fbfcfb;border-radius:8px}#lb svg{width:100%;height:auto;display:block}
.note{color:var(--sub);font-size:13px}
"""

PAGE_JS = """
var lb=document.getElementById('lb'),box=lb.firstElementChild;
document.querySelectorAll('figure').forEach(function(f){f.addEventListener('click',function(){
box.innerHTML='';box.appendChild(f.querySelector('svg').cloneNode(true));lb.classList.add('on');});});
lb.addEventListener('click',function(){lb.classList.remove('on');});
document.addEventListener('keydown',function(e){if(e.key==='Escape')lb.classList.remove('on');});
"""



TAB_CSS = """
.tabs{display:flex;gap:8px;margin:0 0 14px;flex-wrap:wrap}.tabs button{border:1px solid var(--line);background:var(--card);color:var(--ink);
font:inherit;font-size:15px;padding:8px 14px;border-radius:8px;cursor:pointer}.tabs button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.lv{display:none}.lv.on{display:block}td.sub{color:var(--sub);font-size:13px}
"""

TAB_JS = """
document.querySelectorAll('.tabs button').forEach(function(b){b.addEventListener('click',function(){
document.querySelectorAll('.tabs button').forEach(function(x){x.classList.toggle('on',x===b);});
document.querySelectorAll('.lv').forEach(function(x){x.classList.toggle('on',x.id===b.dataset.lv);});});});
"""

LV_NAME = {'none': 'DRC not relaxed', 'abstract': 'DRC relaxed (abstract)', 'spacing': 'DRC relaxed (spacing)', 'all': 'DRC relaxed (all)'}


def lv_name(lv):
    return LV_NAME.get(lv, 'DRC ' + lv)


def drc_text(A):
    lv = A.get('drc_relax', 'none'); n = A.get('tolerated_violations', 0)
    if lv == 'none':
        return 'every reroute kept adds no new DRC violation (no relaxation)'
    what = {'abstract': 'clashes with other cells\' LEF pin / OBS abstracts', 'spacing': 'abstract clashes and spacing violations',
            'all': 'any violation'}.get(lv, lv)
    return 'relaxed (--drc-relax %s: %s allowed); %d such violation%s in the final layout' % (lv, what, n, '' if n == 1 else 's')


def metric_rows(A, unit):
    """[(name, before, after, change text, is an improvement)] of one final layout"""
    K = A['key_metrics']; r, t = K['rerouted_nets'], K['target_nets']; b, f = A['baseline'], A['final']
    return [('coupling of the rerouted nets (%s)' % unit, r['cc_before_fF'], r['cc_after_fF']),
            ('coupling of the %d target nets (%s)' % (t['n'], unit), t['cc_before_fF'], t['cc_after_fF']),
            ('whole-design coupling (%s)' % unit, b['cc_fF'], f['cc_fF']),
            ('wire length (um)', b['wl_um'], f['wl_um']),
            ('vias', b['vias'], f['vias'])]


def num(x):
    return '%d' % x if isinstance(x, int) else '%.2f' % x


def change(name, x, y):
    return ('%+d' % (y - x)) if isinstance(x, int) else fmt_pct(y, x)


def write_page(path, case, res, unit, show_rankers):
    """res: [(level, A, R, figs, overview svg)]"""
    H = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>Coupling reroute demo</title><style>%s%s</style></head><body><main>' % (PAGE_CSS, TAB_CSS),
         '<h1>LLM-guided reroute: %s</h1>' % html.escape(case),
         '<p class="lead">For the nets with the most coupling, candidate Steiner trees are ranked by the LLM, rerouted on the '
         'real layout and measured; the final layout keeps the reroutes that lower coupling without breaking other nets. '
         'Shown for %s.</p>' % ' and '.join(lv_name(lv) for lv, _, _, _, _ in res)]
    kp = []
    for lv, A, _, _, _ in res:
        K = A['key_metrics']
        kp += [('%+.1f%%' % A['cc_change_pct'], 'whole-design coupling, %s' % lv_name(lv)),
               ('%+.1f%%' % K['target_nets']['change_pct'], 'coupling of the %d target nets, %s' % (K['target_nets']['n'], lv_name(lv)))]
    H.append('<div class="kpis">' + ''.join('<div class="kpi"><b>%s</b><span>%s</span></div>' % (v, html.escape(k)) for v, k in kp) + '</div>')
    # comparison table: before | per level: after, change
    H.append('<section><h2>Before / after</h2><table><tr><th>metric</th><th>before</th>'
             + ''.join('<th>after: %s</th><th>change</th>' % html.escape(lv_name(lv)) for lv, _, _, _, _ in res) + '</tr>')
    rows = [metric_rows(A, unit) for _, A, _, _, _ in res]
    for i, (name, x, _) in enumerate(rows[0]):
        cells = ''
        for rr in rows:
            n2, x2, y2 = rr[i]
            good = y2 < x2 and 'coupling' in n2
            pre = ('%s &#8594; ' % num(x2)) if i == 0 else ''   # the rerouted nets differ per level
            cells += '<td>%s%s</td><td%s>%s</td>' % (pre, num(y2), ' class="g"' if good else '', change(n2, x2, y2))
        H.append('<tr><td>%s</td><td>%s</td>%s</tr>' % (html.escape(name), '' if i == 0 else num(x), cells))
    H.append('<tr><td>nets rerouted</td><td></td>' + ''.join('<td>%d</td><td></td>' % A['accepted'] for _, A, _, _, _ in res) + '</tr>')
    H.append('<tr><td>new DRC violations tolerated</td><td></td>'
             + ''.join('<td>%d</td><td></td>' % A.get('tolerated_violations', 0) for _, A, _, _, _ in res) + '</tr>')
    H.append('</table>')
    H.append(''.join('<p class="note"><b>%s:</b> %s</p>' % (html.escape(lv_name(lv)), html.escape(drc_text(A))) for lv, A, _, _, _ in res))
    H.append('</section>')
    # one tab per level: rerouted nets, layout, per-net figures
    H.append('<div class="tabs">' + ''.join('<button data-lv="lv_%s"%s>%s</button>' % (lv, ' class="on"' if k == 0 else '', html.escape(lv_name(lv)))
                                            for k, (lv, _, _, _, _) in enumerate(res)) + '</div>')
    for k, (lv, A, R, figs, ov) in enumerate(res):
        H.append('<div class="lv%s" id="lv_%s">' % (' on' if k == 0 else '', lv))
        if A['nets']:
            H.append('<section><h2>Rerouted nets: %s</h2><table><tr><th>net</th><th>coupling before</th><th>after</th>'
                     '<th>change</th><th>tree</th></tr>' % html.escape(lv_name(lv)))
            for x in A['nets']:
                H.append('<tr><td>%s</td><td>%.2f</td><td>%.2f</td><td class="g">%s</td><td>%s</td></tr>'
                         % (html.escape(clean(x['net'])), x['cc_before_fF'], x['cc_after_fF'], fmt_pct(x['cc_after_fF'], x['cc_before_fF']),
                            html.escape(x['candidate_id'].split(':')[-1])))
            H.append('</table></section>')
        rk = {q: R['rankers'][q] for q in ('llm', 'heuristic', 'random') if q in (R.get('rankers') or {})} if show_rankers else {}
        if rk:
            H.append('<section><h2>Ranking the same candidates</h2><table><tr><th>ranker</th><th>coupling of the nets</th><th>change</th></tr>')
            for q, x in rk.items():
                H.append('<tr><td>%s</td><td>%.2f &#8594; %.2f</td><td class="g">%+.1f%%</td></tr>' % (html.escape(q), x['cc_before'], x['cc_after'], x['change_pct']))
            H.append('</table><p class="note">Each ranker\'s top tree per net, rerouted and measured alone.</p></section>')
        if ov:
            H.append('<section><h2>Layout: %s</h2><figure>%s<figcaption>The area around the rerouted nets (coloured). Click to '
                     'enlarge, hover for net and layer.</figcaption></figure></section>' % (html.escape(lv_name(lv)), ov))
        for name, svg in figs:
            H.append('<section><figure>%s<figcaption>%s (%s): shaded red where the net runs parallel to another net on the same '
                     'layer (within 2 routing pitches). Click to enlarge.</figcaption></figure></section>'
                     % (svg, html.escape(clean(name)), html.escape(lv_name(lv))))
        if not A['nets']:
            H.append('<section><p>No reroute kept at this level.</p></section>')
        H.append('</div>')
    if unit != 'fF':
        H.append('<p class="note">Coupling here is the geometric proxy: um of same-layer parallel run within 2 routing pitches '
                 '(no extraction rules file for this technology).</p>')
    H.append('</main><div id="lb"><div></div></div><script>%s%s</script></body></html>' % (PAGE_JS, TAB_JS))
    path.write_text('\n'.join(H) + '\n')


# ----------------------------------------------------------------------------------------------- main

def final_geo_path(A):
    acc = [s for s in A['steps'] if s.get('accepted')]
    return Path(acc[-1]['dir']) / 'geo.json' if acc else None


def snapshot(out, lv):
    """keep this level's final layout results (the next level overwrites s6_apply / report)"""
    import shutil
    d = out / 'demo' / lv; d.mkdir(parents=True, exist_ok=True)
    for f in ('s6_apply/apply.json', 'report.json'):
        if (out / f).exists():
            shutil.copyfile(str(out / f), str(d / Path(f).name))
    A = json.loads((out / 's6_apply' / 'apply.json').read_text())
    g = final_geo_path(A)
    if g and g.exists():
        shutil.copyfile(str(g), str(d / 'final_geo.json'))
    elif (d / 'final_geo.json').exists():
        (d / 'final_geo.json').unlink()
    for x in ('final.def', 'final.spef'):
        if (out / 's6_apply' / x).exists():
            shutil.copyfile(str(out / 's6_apply' / x), str(d / x))


def load_level(out, lv):
    d = out / 'demo' / lv
    if not (d / 'apply.json').exists():
        return None
    A = json.loads((d / 'apply.json').read_text())
    R = json.loads((d / 'report.json').read_text()) if (d / 'report.json').exists() else {}
    G = json.loads((d / 'final_geo.json').read_text()) if (d / 'final_geo.json').exists() else None
    return A, R, G


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], usage='python3 flow/demo.py A.def TECH.lef CELLS.lef [options]')
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--no-run', action='store_true', help='only draw from an existing output folder')
    ap.add_argument('--draw', dest='max_nets', type=int, default=8, help='draw at most this many rerouted nets per level (default 8)')
    ap.add_argument('--rankers', action='store_true', help='also show the LLM / heuristic / random comparison')
    ap.add_argument('--levels', default='none,abstract',
                    help='DRC levels to show, comma-separated (default none,abstract: not relaxed, and clashes with cell '
                         'LEF abstracts tolerated); --drc-relax X shows only X')
    ap.add_argument('--drc-relax', dest='one_level', default=None)
    ap.add_argument('--out', default=None)
    a, rest = ap.parse_known_args()
    levels = [a.one_level] if a.one_level else [x.strip() for x in a.levels.split(',') if x.strip()]
    bad = [x for x in levels if x not in LV_NAME]
    if bad:
        print('unknown DRC level(s): %s (use none, abstract, spacing, all)' % ', '.join(bad)); sys.exit(2)
    first = a.inputs[0]
    case = re.sub(r'\.def(\.gz)?$', '', Path(first).name, flags=re.I) if first.lower().endswith(('.def', '.def.gz')) else first
    out = Path(a.out).resolve() if a.out else REPO / 'out' / case
    print('LLM-guided coupling reroute: %s' % case)
    if not a.no_run:
        order = list(levels)
        try:   # the level the output folder holds now goes first (nothing to redo for it)
            cur = json.loads((out / 'selection.json').read_text()).get('drc_relax')
            if cur in order:
                order.remove(cur); order.insert(0, cur)
        except (OSError, ValueError):
            pass
        for k, lv in enumerate(order):
            print('\n%s (%d/%d)' % (lv_name(lv), k + 1, len(order)))
            argv = a.inputs + rest + ['--drc-relax', lv] + (['--out', str(out)] if a.out else [])
            if run_flow(argv, out / 'demo' / ('run_%s.log' % lv)):
                sys.exit(1)
            snapshot(out, lv)
    res = []
    for lv in levels:
        got = load_level(out, lv)
        if not got:
            print('  no result for %s in %s yet: run without --no-run first' % (lv_name(lv), out)); continue
        if 'key_metrics' not in got[0]:
            print('  the %s result was written by an older flow: run without --no-run once' % lv_name(lv)); continue
        res.append((lv,) + got)
    if not res:
        sys.exit(1)
    unit = 'fF' if (out / 'inputs.json').exists() and json.loads((out / 'inputs.json').read_text()).get('rules_sha256') != 'proxy' else 'um'
    # screen: one column per level
    W = 34
    print('\nResult (coupling%s)' % (' in fF' if unit == 'fF' else ': um of same-layer parallel run'))
    print('  %-16s' % '' + ''.join('%-*s' % (W, lv_name(lv)) for lv, _, _, _ in res))
    rows = [metric_rows(A, unit) for _, A, _, _ in res]
    short = ['rerouted nets', 'target nets', 'whole design', 'wire length', 'vias']
    for i, lab in enumerate(short):
        cells = ''
        for rr in rows:
            n2, x2, y2 = rr[i]
            cells += '%-*s' % (W, '%s -> %s  %s' % (num(x2), num(y2), change(n2, x2, y2)))
        print('  %-16s%s' % (lab, cells))
    print('  %-16s' % 'nets rerouted' + ''.join('%-*s' % (W, A['accepted']) for _, A, _, _ in res))
    print('  %-16s' % 'DRC tolerated' + ''.join('%-*s' % (W, '%d new violation(s)' % A.get('tolerated_violations', 0) if A.get('drc_relax') != 'none'
                                                          else 'none (no new DRC)') for _, A, _, _ in res))
    for lv, A, R, _ in res:
        if A['nets']:
            print('\nRerouted nets, %s' % lv_name(lv))
            for x in A['nets']:
                print('  %-24s %8.2f -> %8.2f   %s' % (clean(x['net'])[:24], x['cc_before_fF'], x['cc_after_fF'],
                                                   fmt_pct(x['cc_after_fF'], x['cc_before_fF'])))
        if a.rankers and R.get('rankers'):
            print('  ranking the same candidates (top tree per net, each rerouted alone):')
            for q in ('llm', 'heuristic', 'random'):
                if q in R['rankers']:
                    x = R['rankers'][q]
                    print('    %-10s %9.2f -> %9.2f   %+.1f%%' % (q, x['cc_before'], x['cc_after'], x['change_pct']))
    # figures
    before = json.loads((out / 's1_import' / 'geo.json').read_text())
    for old in (out / 'demo').glob('*.svg'):   # pictures of an earlier one-level demo
        old.unlink()
    page = []
    for lv, A, R, after in res:
        d = out / 'demo' / lv; figs = []; ov = ''
        for old in d.glob('*.svg'):
            old.unlink()
        if after:
            nets = [x['net'] for x in sorted(A['nets'], key=lambda x: -(x['cc_before_fF'] - x['cc_after_fF']))][:max(a.max_nets, 0)]
            ov = overview_figure(before, after, nets)
            (d / 'overview.svg').write_text(ov + '\n')
            recs = {x['net']: dict(x, _unit=unit, _policy=A.get('policy', 'llm')) for x in A['nets']}
            for i, n in enumerate(nets):
                if n not in before['nets'] or n not in after['nets']:
                    continue
                svg = net_figure(before, after, n, recs[n], i, len(nets))
                figs.append((n, svg))
                (d / ('net_%02d_%s.svg' % (i + 1, re.sub(r'[^A-Za-z0-9_.-]+', '_', clean(n))))).write_text(svg + '\n')
        page.append((lv, A, R, figs, ov))
    idx = out / 'demo' / 'index.html'
    write_page(idx, case, page, unit, a.rankers)
    print('\nPictures: %s' % idx)
    print('  open:  firefox %s &' % idx)
    print('  (also as SVG files, and each level\'s final.def, in %s/<level>/)' % (out / 'demo'))


if __name__ == '__main__':
    main()
