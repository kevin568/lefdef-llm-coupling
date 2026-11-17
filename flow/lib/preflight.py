"""Preflight: let OpenROAD read the LEFs (and the DEF) once, and repair what it rejects, before any stage runs.

OpenROAD discards a whole LEF when it reports errors in it (ODB-0289), e.g. a NONDEFAULTRULE that uses a via or a via
rule whose definition it could not build (USEVIA subTap / USEVIARULE M1_LiAct on device layers). For each LEF in order,
the preflight reads it; on errors it writes a copy (out/.cache/) without the reported vias / via rules: their
definitions and every USEVIA / USEVIARULE line naming them. It repeats until OpenROAD accepts the file (at most 6
rounds). Errors it cannot repair are all listed at once. Original files are never changed."""
import os
import re
import subprocess
import tempfile
from pathlib import Path

from . import common as C

CACHE = Path(os.environ.get('LEFDEF_CACHE') or C.REPO / 'out' / '.cache')
READ_TCL = r'''
set n 0
foreach l $::env(LEFS) {
  incr n
  if {[catch {read_lef $l} e]} { puts "FLOW_LEF_FAIL $n $e" }
}
if {[info exists ::env(DEF_IN)] && $::env(DEF_IN) != ""} {
  if {[catch {read_def $::env(DEF_IN)} e]} { puts "FLOW_DEF_FAIL $e" }
}
puts "FLOW_DONE"
'''


def read_with_openroad(lefs, def_in=''):
    """OpenROAD log text of reading the LEFs (and DEF)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tcl = CACHE / 'preflight_read.tcl'
    tcl.write_text(READ_TCL)
    env = dict(os.environ, LEFS=' '.join(map(str, lefs)), DEF_IN=str(def_in))
    r = subprocess.run([C.openroad_bin(), '-no_init', '-no_splash', '-exit', str(tcl)], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
    return r.stdout.decode(errors='replace')


def _errors(log):
    return [l.strip() for l in log.splitlines() if re.search(r'\berror\b|\[ERROR', l, re.I) and 'FLOW_' not in l]


def _undefined_names(log):
    vias = set(re.findall(r'undefined VIA (?!GENERATE RULE)(\S+)', log))          # USEVIA of a via not built
    vias |= set(re.findall(r'in VIA \(([^)\s]+)\)', log))                          # ODB-0209: via on an unknown layer
    rules = set(re.findall(r'undefined VIA GENERATE RULE (\S+)', log))             # USEVIARULE of a rule not built
    rules |= set(re.findall(r'VIARULE \(([^)\s]+)\)', log))                        # ODB-0218: rule on an unknown layer
    return vias, rules


def _strip(text, vias, rules):
    """LEF text without the VIA / VIARULE blocks of these names and without USEVIA / USEVIARULE lines naming them"""
    lines = text.splitlines(); out = []; i = 0
    names = vias | rules
    while i < len(lines):
        m = re.match(r'^\s*(VIA|VIARULE)\s+(\S+)(?:\s+(?:DEFAULT|GENERATE)(?:\s+DEFAULT)?)?\s*$', lines[i])
        if m and m.group(2) in names:
            end = re.compile(r'^\s*END\s+' + re.escape(m.group(2)) + r'\s*$')
            j = i + 1
            while j < len(lines) and not end.match(lines[j]):
                j += 1
            i = j + 1
            continue
        u = re.match(r'^\s*(USEVIA|USEVIARULE)\s+(\S+)\s*;', lines[i])
        if u and u.group(2) in names:
            i += 1
            continue
        out.append(lines[i]); i += 1
    return '\n'.join(out) + '\n'


def fix_lefs(lefs):
    """-> (LEF paths OpenROAD accepts, notes). Reads each LEF in order (with the ones before it), repairs on errors."""
    lefs = [str(l) for l in lefs]; notes = []
    for k in range(len(lefs)):
        cur = lefs[k]; removed_v, removed_r = set(), set()
        for _ in range(6):
            log = read_with_openroad(lefs[:k] + [cur])
            fail = [l for l in log.splitlines() if l.startswith('FLOW_LEF_FAIL %d ' % (k + 1))]
            if not fail:
                break
            vias, rules = _undefined_names(log)
            vias -= removed_v; rules -= removed_r
            if not vias and not rules:
                errs = _errors(log)
                if k > 0:   # a cell LEF: leave it out (its cells are then left out of the DEF copy) and keep going
                    notes.append(f'{Path(lefs[k]).name}: OpenROAD rejects it ({errs[0] if errs else "error"}); left out')
                    cur = None
                    break
                raise C.StageError(f'OpenROAD rejects the technology LEF {Path(lefs[k]).name} and the errors are not ones '
                                   f'the flow can repair. All {len(errs)} error line(s):\n    ' + '\n    '.join(errs[:40]))
            removed_v |= vias; removed_r |= rules
            new = CACHE / C.sha256(lefs[k])[:16] / (Path(lefs[k]).stem + '.openroad.lef')
            new.parent.mkdir(parents=True, exist_ok=True)
            new.write_text(_strip(Path(cur).read_text(errors='replace'), vias, rules))
            cur = str(new)
        else:
            raise C.StageError(f'OpenROAD still rejects {Path(lefs[k]).name} after 6 repair rounds; errors:\n    '
                               + '\n    '.join(_errors(log)[:40]))
        if cur is not None and cur != lefs[k]:
            names = sorted(removed_v) + sorted(removed_r)
            notes.append(f'{Path(lefs[k]).name}: OpenROAD could not use {len(names)} via / via rule definition(s) '
                         f'({", ".join(names[:6])}{" ..." if len(names) > 6 else ""}); a copy without them is used')
        lefs[k] = cur
    return [l for l in lefs if l], notes


def check_def(lefs, def_in):
    """None when OpenROAD reads the DEF; otherwise the list of its error lines."""
    log = read_with_openroad(lefs, def_in)
    if any(l.startswith('FLOW_DEF_FAIL') for l in log.splitlines()) or 'DEF parser returns an error' in log:
        return _errors(log)
    return None


# ---------------------------------------------------------------- detailed router (TritonRoute) preflight
def _tech_copy(path, text, tag):
    new = CACHE / C.sha256(path)[:16] / (Path(path).stem + '.' + tag + '.lef')
    new.parent.mkdir(parents=True, exist_ok=True)
    new.write_text(text)
    return str(new)


def _vias_on_layers(text, layers):
    """names of the VIA / VIARULE blocks that use one of these layers"""
    out = set()
    for m in re.finditer(r'^\s*(VIA|VIARULE)\s+(\S+)[^\n]*\n(.*?)^\s*END\s+(\S+)\s*$', text, re.M | re.S):
        if m.group(4) == m.group(2) and any(re.search(r'^\s*LAYER\s+' + re.escape(l) + r'\s*;', m.group(3), re.M)
                                             for l in layers):
            out.add(m.group(2))
    return out


def unmark_mimcap(lefs):
    """The detailed router ignores LEF58_TYPE MIMCAP layers (e.g. a MIM capacitor top metal) and then stops on every
    via / via rule that uses one (DRT-0129 "Unknown layer CMT for viarule MT_CMT"); leaving those vias out instead
    leaves their cut layer without vias (DRT-0233). So a copy of the technology LEF drops only the MIMCAP mark: the
    layer is an ordinary metal layer for the router (signal routing still stops below it). -> (lefs, notes)"""
    lefs = [str(l) for l in lefs]
    text = Path(lefs[0]).read_text(errors='replace')
    mim = []

    def unmark(m):
        body = re.sub(r'^[ \t]*PROPERTY[ \t]+LEF58_TYPE[ \t]+"[^"]*\bMIMCAP\b[^"]*"[ \t]*;[ \t]*\r?\n', '', m.group(2), flags=re.M)
        if body != m.group(2):
            mim.append(m.group(1))
        return m.group(0).replace(m.group(2), body, 1)
    new = re.sub(r'^[ \t]*LAYER[ \t]+(\S+)[ \t]*\r?\n(.*?)^[ \t]*END[ \t]+\1\b', unmark, text, flags=re.M | re.S)
    if not mim:
        return lefs, [], []
    lefs[0] = _tech_copy(lefs[0], new, 'router')
    return lefs, mim, [f'MIM capacitor layer(s) {", ".join(mim)}: the detailed router ignores such layers and stops on '
                  'their vias, so the technology LEF copy drops the MIMCAP mark (no signal routing on them)']


LAST_PROBE_LOG = None


def router_probe(lefs, def_in, layers, skip_tracks=(), route_net=''):
    """Run the reroute script once with an empty guide (nothing is rerouted): the detailed router reads the whole
    technology and does pin access for every cell, which is where it stops on what it cannot use. -> error lines.
    route_net: instead really reroute this one net (global + detailed route), for designs without a gcell grid."""
    d = CACHE / ('router_probe' + ('_' + '_'.join(skip_tracks) if skip_tracks else '') + '_' + C.sha256(lefs[0])[:6]
                 + ('_net' if route_net else ''))
    d.mkdir(parents=True, exist_ok=True)
    (d / 'empty.guide').write_text('')
    log = d / 'route.log'
    global LAST_PROBE_LOG
    LAST_PROBE_LOG = log   # tools/diagnose.py prints what the router logged last
    try:
        C.run_openroad(C.OR_DIR / 'route.tcl', dict(LEFS=' '.join(map(str, lefs)), STATE_DEF=def_in, OUT_DEF=d / 'out.def',
                       LAYERS=layers, TARGETS='{' + route_net + '}' if route_net else '',
                       MODE='default' if route_net else 'probe', GUIDE=d / ('grt.guide' if route_net else 'empty.guide'),
                       DRC=d / 'drc.rpt', SKIP_TRACKS=' '.join(skip_tracks)), log, timeout=3600)
        return []
    except C.StageError:
        return [e for e in _errors(log.read_text(errors='replace')) if 'GRT-0253' not in e] or ['(no error line: see ' + str(log) + ')']


def fix_router(lefs, def_in, layers, skip_tracks=()):
    """Probe the detailed router; vias / via rules it rejects (DRT-0129 unknown layer) are left out of a copy of the
    technology LEF, together with every other via on that layer, and the probe repeats (at most 8 rounds).
    -> (lefs, notes, remaining error lines or [])"""
    lefs = [str(l) for l in lefs]; removed = set(); errs = []
    for _ in range(8):
        errs = router_probe(lefs, def_in, layers, skip_tracks)
        if not errs:
            break
        text = Path(lefs[0]).read_text(errors='replace')
        bad = re.findall(r'Unknown layer (\S+) for (?:viarule|via|VIA|VIARULE) (\S+?)\.?\s*$', '\n'.join(errs), re.M)
        names = ({v for _, v in bad} | _vias_on_layers(text, {l for l, _ in bad})) - removed
        if not names:
            break
        removed |= names
        lefs[0] = _tech_copy(lefs[0], _strip(text, names, names), 'router')
    notes = [f'the detailed router rejects {len(removed)} via / via rule(s) ({", ".join(sorted(removed)[:4])}'
             f'{" ..." if len(removed) > 4 else ""}): left out of the technology LEF copy'] if removed else []
    return lefs, notes, errs


# ---------------------------------------------------------------- MIM capacitor layers: let the router decide
TECH_TCL = r'''
foreach l $::env(LEFS) { if {[catch {read_lef $l} e]} { puts "FLOW_LEF_FAIL $e" } }
if {[catch {read_def $::env(DEF_IN)} e]} { puts "FLOW_DEF_FAIL $e" }
set f [open $::env(GUIDE) w]; close $f
catch {read_guides $::env(GUIDE)}
if {[catch {detailed_route -verbose 0} e]} { puts "FLOW_DRT_FAIL $e" }
puts FLOW_DONE
'''


def tech_probe(lefs):
    """technology errors of the detailed router for these LEFs (it parses the technology before anything else),
    using a tiny empty design: seconds. -> '[ERROR DRT-...' lines"""
    d = CACHE / 'tech_probe'; d.mkdir(parents=True, exist_ok=True)
    u = re.search(r'DATABASE\s+MICRONS\s+(\d+)', Path(lefs[0]).read_text(errors='replace'))
    u = int(u.group(1)) if u else 1000
    (d / 'probe.def').write_text(f'VERSION 5.8 ;\nDIVIDERCHAR "/" ;\nBUSBITCHARS "[]" ;\nDESIGN techprobe ;\n'
                                 f'UNITS DISTANCE MICRONS {u} ;\nDIEAREA ( 0 0 ) ( {100 * u} {100 * u} ) ;\nEND DESIGN\n')
    (d / 'probe.tcl').write_text(TECH_TCL)
    env = dict(os.environ, LEFS=' '.join(map(str, lefs)), DEF_IN=str(d / 'probe.def'), GUIDE=str(d / 'empty.guide'))
    try:
        r = subprocess.run([C.openroad_bin(), '-no_init', '-no_splash', '-exit', str(d / 'probe.tcl')], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
        log = r.stdout.decode(errors='replace')
    except subprocess.TimeoutExpired:
        return []
    (d / 'probe.log').write_text(log)
    return [l.strip() for l in log.splitlines() if re.search(r'\[ERROR (DRT|ODB)-\d+\]', l)]


def _mim_layers(text):
    return [n for n, b in ((m.group(1), m.group(2)) for m in
            re.finditer(r'^[ \t]*LAYER[ \t]+(\S+)[ \t]*\r?\n(.*?)^[ \t]*END[ \t]+\1\b', text, re.M | re.S))
            if re.search(r'LEF58_TYPE\s+"\s*TYPE\s+MIMCAP', b)]


def _drop_layer_geometry(text, layers):
    """MACRO PIN / OBS geometry on these layers left out ('LAYER x ;' and its shapes up to the next LAYER / END)"""
    out, skip = [], False
    for line in text.splitlines(True):
        m = re.match(r'^\s*LAYER\s+(\S+)\s*;', line)
        if m:
            skip = m.group(1) in layers
            if skip:
                continue
        elif skip and re.match(r'^\s*(RECT|POLYGON|PATH|WIDTH|ITERATE|\(|-?[\d.])', line):
            continue
        else:
            skip = False if re.match(r'^\s*(END|VIA|LAYER|PORT)\b', line) else skip
        out.append(line)
    return ''.join(out)


def mimcap_variants(lefs):
    """LEF58_TYPE MIMCAP layers (MIM capacitor metals): the old detailed router ignores them, which breaks either the
    via rules ON them (DRT-0129 unknown layer) or, with the mark dropped, the via rules that skip OVER them
    (DRT-0126 / DRT-0130 non-consecutive layers), or cell shapes on them (DRT-0094). Which way works depends on the
    PDK and the cells, so the flow setup tries these variants on the real design (router probe) and keeps the first
    one the router accepts:
      1 keep the mark, leave out the vias / via rules on the layer (and build no tracks on it)
      2 drop the mark (an ordinary metal layer for the router)
      3 remove the layer (and the cell geometry on it)
    -> [dict(desc, lefs, mim, skip, tech_errs)], technology-clean first; [] without MIMCAP layers"""
    lefs = [str(l) for l in lefs]
    text = Path(lefs[0]).read_text(errors='replace')
    mim = _mim_layers(text)
    if not mim:
        return []
    on = _vias_on_layers(text, mim)
    unmarked = re.sub(r'^[ \t]*PROPERTY[ \t]+LEF58_TYPE[ \t]+"[^"]*\bMIMCAP\b[^"]*"[ \t]*;[ \t]*\r?\n', '', text, flags=re.M)
    removed = re.sub(r'^[ \t]*LAYER[ \t]+(\S+)[ \t]*\r?\n.*?^[ \t]*END[ \t]+\1\b[^\n]*\n?',
                     lambda m: '' if m.group(1) in mim else m.group(0), _strip(text, on, on), flags=re.M | re.S)
    variants = [(f'keep the MIMCAP mark, leave out the {len(on)} via / via rule(s) on it', _strip(text, on, on), False),
                ('drop the MIMCAP mark', unmarked, False),
                ('remove the layer and the cell geometry on it', _drop_layer_geometry(removed, set(mim)), True)]
    out = []
    for k, (desc, new, cells_too) in enumerate(variants):
        cand = [_tech_copy(lefs[0], new, f'mim{k + 1}')]
        for l in lefs[1:]:
            if cells_too:
                t = Path(l).read_text(errors='replace')
                cand.append(_tech_copy(l, _drop_layer_geometry(t, set(mim)), f'nomim{k + 1}'))
            else:
                cand.append(l)
        # tracks are never built on a layer the router ignores (DRT-0094 "Cannot find layer")
        out.append(dict(desc=desc, lefs=cand, mim=mim, skip=mim if k == 0 else [], tech_errs=tech_probe(cand)))
    return sorted(out, key=lambda v: bool(v['tech_errs']))   # technology-clean variants first, in this order


def pick_mimcap(lefs):
    """the first technology-clean variant (see mimcap_variants) -> (lefs, MIM layer names, notes)"""
    v = mimcap_variants(lefs)
    if not v:
        return [str(l) for l in lefs], [], []
    return v[0]['lefs'], v[0]['mim'], [f'MIM capacitor layer(s) {", ".join(v[0]["mim"])}: "{v[0]["desc"]}"']


# ---------------------------------------------------------------- fixed vias for cut layers with only via rules
def add_default_vias(lefs):
    """The old detailed router needs at least one fixed VIA per cut layer between two metal layers (DRT-0233
    "V1 does not have any vias"); LEFs that give only VIARULE ... GENERATE for a cut layer get, in a copy of the
    technology LEF, one fixed VIA <rule>_FLOW DEFAULT built from that rule: its cut RECT, and the bottom / top
    metal rectangles grown by the rule's ENCLOSURE values. -> (lefs, notes)"""
    lefs = [str(l) for l in lefs]
    t = Path(lefs[0]).read_text(errors='replace')
    types = {m.group(1): (re.search(r'^\s*TYPE\s+(\S+)', m.group(2), re.M) or [None, ''])[1]
             for m in re.finditer(r'^[ \t]*LAYER[ \t]+(\S+)[ \t]*\r?\n(.*?)^[ \t]*END[ \t]+\1\b', t, re.M | re.S)}
    order = list(types)
    blocks = list(re.finditer(r'^[ \t]*(VIA|VIARULE)[ \t]+(\S+)([^\n]*)\n(.*?)^[ \t]*END[ \t]+\2\b[^\n]*\n?', t, re.M | re.S))

    def layers_of(body):
        out, cur = [], None
        for line in body.splitlines():
            m = re.match(r'^\s*LAYER\s+(\S+)\s*;', line)
            if m:
                cur = [m.group(1), []]; out.append(cur)
            elif cur is not None and line.strip():
                cur[1].append(line.strip())
        return out
    fixed_cuts = set()
    for b in blocks:
        if b.group(1) == 'VIA':
            fixed_cuts |= {l for l, _ in layers_of(b.group(4)) if types.get(l) == 'CUT'}
    new, made = [], []
    mim = set(_mim_layers(t))   # rules on a MIM capacitor layer last (the MIM handling may leave them out)
    for b in sorted(blocks, key=lambda b: any(l in mim for l, _ in layers_of(b.group(4)))):
        if b.group(1) != 'VIARULE' or 'GENERATE' not in b.group(3):
            continue
        ls = layers_of(b.group(4))
        cuts = [x for x in ls if types.get(x[0]) == 'CUT']
        mets = [x for x in ls if types.get(x[0]) == 'ROUTING']
        if len(cuts) != 1 or len(mets) != 2 or cuts[0][0] in fixed_cuts:
            continue
        cut = cuts[0]
        r = next((re.findall(r'-?[\d.]+', s) for s in cut[1] if s.startswith('RECT')), None)
        if not r or len(r) < 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in r[:4])
        mets.sort(key=lambda x: order.index(x[0]))
        lines = [f'VIA {b.group(2)}_FLOW DEFAULT']
        for name, stmts in (mets[0], cut, mets[1]):
            lines.append(f'  LAYER {name} ;')
            if name == cut[0]:
                lines.append(f'    RECT {x1:g} {y1:g} {x2:g} {y2:g} ;')
                continue
            e = next((re.findall(r'-?[\d.]+', s) for s in stmts if s.startswith('ENCLOSURE')), ['0', '0'])
            ex, ey = (float(e[0]), float(e[1])) if len(e) >= 2 else (0., 0.)
            lines.append(f'    RECT {x1 - ex:g} {y1 - ey:g} {x2 + ex:g} {y2 + ey:g} ;')
        lines.append(f'END {b.group(2)}_FLOW\n')
        new.append('\n'.join(lines)); made.append(f'{b.group(2)}({cut[0]})'); fixed_cuts.add(cut[0])
    if not new:
        return lefs, []
    k = t.rfind('\nEND LIBRARY')
    k = k if k >= 0 else len(t)
    lefs[0] = _tech_copy(lefs[0], t[:k] + '\n' + '\n'.join(new) + t[k:], 'vias')
    return lefs, [f'{len(made)} cut layer(s) have only via rules, no fixed via (the router needs one: DRT-0233): '
                  f'a fixed via is made from each rule ({", ".join(made[:8])})']
