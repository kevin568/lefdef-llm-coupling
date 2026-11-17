#!/usr/bin/env python3
"""Diagnose a LEF/DEF pair for the flow in ONE run: every check runs, none stops the others.

  python3 tools/diagnose.py A.def tech.lef cells.lef            (same arguments as flow/run.py)
  python3 tools/diagnose.py A.def tech.lef cells.lef --nets 5   (reroute test on 5 nets, default 3)

It reads the inputs raw and after the flow's repairs, lists what the technology / cells / DEF contain that OpenROAD
(the installed version) cannot use, does the DEF write/read round trip, the geometry export, and real one-net
reroutes, prints the key lines of every check, and writes the full detail to out/diagnose/<DEF name>/report.txt.
Everything stays on this machine.
"""
import argparse
import contextlib
import io
import json
import re
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'flow'))
from lib import common as C, preflight as P, defcells as D   # noqa: E402

CHECKS = []


def check(name):
    """decorator: run a check, record (status, summary, details) and any exception; never raises"""
    def deco(fn):
        def run(*a, **k):
            t0 = time.time(); out = io.StringIO()
            rec = dict(name=name, status='OK', summary='', details=[], seconds=0.)
            try:
                with contextlib.redirect_stdout(out):
                    res = fn(*a, **k) or {}
                rec.update(res)
            except Exception as e:   # noqa: BLE001 -- a diagnosis tool reports everything
                rec.update(status='FAIL', summary=f'{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ""}',
                           details=rec['details'] + str(e).splitlines()[1:80] + ['--- traceback ---'] +
                           traceback.format_exc().splitlines()[-12:])
            printed = [l for l in out.getvalue().splitlines() if l.strip()]
            if name.startswith('7'):
                rec.setdefault('key', [])
                rec['key'] = [l.strip() for l in printed if re.match(r'\s*(note|warning|--layers|--rules|\[ERROR|ERROR|Error|\[WARNING)', l)] + rec['key']
            if rec['status'] == 'FAIL':
                rec['key'] = [rec['summary']] + rec.get('key', [])
            if printed:
                rec['details'] = ['--- flow output ---'] + printed + rec['details']
            rec['seconds'] = round(time.time() - t0, 1)
            CHECKS.append(rec)
            sys.stdout.write(f'[{rec["status"]:4s}] {name}: {rec["summary"][:150]}\n'); sys.stdout.flush()
            return rec, res if rec['status'] != 'FAIL' else None
        return run
    return deco


def layer_blocks(text):
    for m in re.finditer(r'^[ \t]*LAYER[ \t]+(\S+)[ \t]*\r?\n(.*?)^[ \t]*END[ \t]+\1\b', text, re.M | re.S):
        yield m.group(1), m.group(2)


def val(body, key):
    m = re.search(r'^\s*' + key + r'\s+([^;\n]+?)\s*;', body, re.M)
    return m.group(1).strip() if m else ''


def section(text, name):
    s = D._section(text, name)
    return s[1] if s else ''


def norm(e):
    return re.sub(r'(?<![-\w.])\d+(\.\d+)?\b', '#', e.strip())


def distinct(lines, n=40):
    seen = {}
    for l in lines:
        seen[norm(l)] = seen.get(norm(l), 0) + 1
    order = lambda kv: ('ERROR' not in kv[0][:20], -kv[1])   # real errors first, then the most frequent warnings
    return [f'{c:4d}x {l[:220]}' for l, c in sorted(seen.items(), key=order)[:n]]


# ------------------------------------------------------------------------------------------------ checks
@check('1 environment')
def c_env():
    exe = C.openroad_bin()
    tcl = P.CACHE / 'diag_version.tcl'; P.CACHE.mkdir(parents=True, exist_ok=True)
    tcl.write_text('puts "VERSION [ord::openroad_version]"\nputs FLOW_DONE\n')
    log = P.CACHE / 'diag_version.log'
    C.run_openroad(tcl, {}, log)
    v = re.search(r'VERSION (\S+)', log.read_text(errors='replace'))
    return dict(summary=f'python {sys.version.split()[0]}, OpenROAD {v.group(1) if v else "?"} ({exe})')


@check('2 LEF files (raw OpenROAD read)')
def c_lef_raw(lefs):
    log = P.read_with_openroad(lefs)
    errs = P._errors(log)
    info = re.findall(r'unsupported (\S+) property', log)
    det = [f'file {i + 1}: {l}' for i, l in enumerate(lefs)]
    if info:
        det.append(f'{len(info)} unsupported LEF58 / tech properties (INFO only, ignored by OpenROAD): '
                   + ', '.join(f'{k} x{info.count(k)}' for k in sorted(set(info))))
    det += ['errors:'] + distinct(errs) if errs else []
    return dict(status='WARN' if errs else 'OK', details=det, key=distinct(errs, 6),
                summary=f'{len(errs)} error line(s) when OpenROAD reads the LEF files as given' if errs else 'OpenROAD reads them as given')


@check('3 technology inventory')
def c_tech(tech, top):
    t = Path(tech).read_text(errors='replace')
    units = val(t, r'DATABASE\s+MICRONS') or val(t, 'DATABASE MICRONS')
    grid = val(t, 'MANUFACTURINGGRID')
    rows, issues = [], []
    names = []
    for name, body in layer_blocks(t):
        typ = val(body, 'TYPE')
        if typ not in ('ROUTING', 'CUT', 'MASTERSLICE', 'OVERLAP', 'IMPLANT') and not typ:
            continue
        names.append(name)
        mim = bool(re.search(r'LEF58_TYPE\s+"\s*TYPE\s+MIMCAP', body))
        d, p, w, s = val(body, 'DIRECTION'), val(body, 'PITCH'), val(body, 'WIDTH'), val(body, 'SPACING')
        rows.append(f'{name:12s} {typ:11s} dir={d or "-":10s} pitch={p or "-":12s} width={w or "-":8s} '
                    f'spacing={s or "-":8s}{"  MIMCAP" if mim else ""}')
        if typ == 'ROUTING':
            if not p or all(float(x) == 0 for x in p.split()):
                issues.append(f'routing layer {name} has no PITCH (flow: tracks from width+spacing; signal routing stops below it)')
            if not d:
                issues.append(f'routing layer {name} has no DIRECTION (global router error GRT-0084 if it is a signal layer)')
            if mim:
                issues.append(f'routing layer {name} is LEF58_TYPE MIMCAP (detailed router ignores it: flow drops the mark in its copy)')
    unknown = []
    for m in re.finditer(r'^\s*(VIA|VIARULE)\s+(\S+)[^\n]*\n(.*?)^\s*END\s+\2\s*$', t, re.M | re.S):
        bad = sorted({l for l in re.findall(r'^\s*LAYER\s+(\S+)\s*;', m.group(3), re.M) if l not in names})
        if bad:
            unknown.append(f'{m.group(1)} {m.group(2)} uses layer(s) not defined in this LEF: {", ".join(bad)}')
    rl = C.lef_routing_layers([tech])
    det = [f'units {units or "?"} DBU/um, manufacturing grid {grid or "?"} um, routing layers {", ".join(rl)}, '
           f'top signal layer used by the flow: {top}', '--- layers ---'] + rows
    if issues:
        det += ['--- layer issues ---'] + issues
    if unknown:
        det += [f'--- {len(unknown)} via / via rule(s) on undefined (device) layers (flow leaves them out) ---'] + unknown[:40]
    return dict(status='WARN' if issues or unknown else 'OK', details=det, key=issues + unknown[:4],
                summary=f'{len(rl)} routing layers, {len(issues)} layer issue(s), {len(unknown)} via(s) on undefined layers')


@check('4 DEF inventory')
def c_def(def_path, lefs, top):
    t = Path(def_path).read_text(errors='replace')
    det = []
    secs = re.findall(r'^\s*([A-Z]+)\s+(\d+)\s*;', t, re.M)
    det.append('sections: ' + ', '.join(f'{a} {b}' for a, b in secs if a not in ('VERSION', 'UNITS')))
    issues = []
    nm = len(re.findall(r'\bMASK\s+[0-9A-Fa-f]+\b', t))
    if nm:
        issues.append(f'{nm} multi-patterning MASK marks (old OpenROAD writes them back unreadable: flow drops them)')
    for s in D.NAMED:
        body = section(t, s)
        names = re.findall(r'^\s*-\s+(\S+)', body, re.M)
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            issues.append(f'{s}: {len(dup)} name(s) defined more than once (e.g. {", ".join(dup[:3])}): flow keeps the first')
    nets = section(t, 'NETS')
    stm = [x for x in re.split(r';\s*\n', nets) if x.strip().startswith('-')]
    pg = [x for x in stm if re.search(r'USE\s+(POWER|GROUND)', x)]
    fixed = [x for x in stm if re.search(r'\+\s*(FIXED|COVER)\s', x)]
    routed = [x for x in stm if re.search(r'\+\s*ROUTED\s', x)]
    conn = lambda x: len(re.findall(r'\(\s*\S+\s+\S+\s*\)', x.split('+')[0]))
    sig = [x for x in stm if x not in pg and re.search(r'\+\s*(ROUTED|FIXED|COVER)\s', x)]
    h = {'0-1': 0, '2-8': 0, '>8': 0}
    for x in sig:
        c = conn(x); h['0-1' if c < 2 else '2-8' if c <= 8 else '>8'] += 1
    det.append(f'NETS: {len(stm)} nets, {len(routed)} with ROUTED, {len(fixed)} with FIXED/COVER wiring, '
               f'{len(pg)} POWER/GROUND nets inside regular NETS, signal nets with wiring by connections: {h}')
    if pg:
        issues.append(f'{len(pg)} POWER/GROUND net(s) in regular NETS (detailed router DRT-0305: flow routes them as signal, never rips them)')
    if fixed:
        issues.append(f'{len(fixed)} net(s) with FIXED/COVER wiring (flow writes them as ROUTED so they can be rerouted)')
    if sig and not h['2-8']:
        issues.append('no signal net has 2..8 connections (multi-finger devices): flow takes every net with >= 2')
    if not sig:
        issues.append('no signal net has wiring in NETS: nothing to reroute (is the routing in SPECIALNETS?)')
    ndr = section(t, 'NONDEFAULTRULES')
    if ndr:
        ndr_names = re.findall(r'^\s*-\s+(\S+)', ndr, re.M)
        det.append('NONDEFAULTRULES: ' + ', '.join(ndr_names))
    pins = section(t, 'PINS')
    play = re.findall(r'\+\s*LAYER\s+(\S+)', pins)
    n_pins = len(re.findall(r'^\s*-\s+', pins, re.M))
    det.append(f'PINS: {n_pins} I/O pins, shapes on layers ' + ', '.join(f'{l} x{play.count(l)}' for l in sorted(set(play))))
    if not re.search(r'^TRACKS\s', t, re.M):
        det.append('no TRACKS in the DEF (flow makes them from the LEF pitches)')
    comp = section(t, 'COMPONENTS')
    masters = re.findall(r'^\s*-\s+\S+\s+(\S+)', comp, re.M)
    macros = D.lef_macros(lefs)
    miss = sorted(set(masters) - macros)
    det.append('cell types used: ' + ', '.join(f'{m} x{masters.count(m)}' for m in sorted(set(masters))))
    if miss:
        issues.append(f'{len(miss)} cell type(s) not defined in the LEF files: {", ".join(miss[:8])} (flow leaves them out)')
    return dict(status='WARN' if issues else 'OK', details=det + (['--- issues (repaired by the flow) ---'] + issues if issues else []),
                key=issues, summary=f'{len(stm)} nets ({len(sig)} wired signal), {len(masters)} instances, {len(issues)} item(s) the flow repairs')


@check('5 cell pins the router can reach')
def c_pins(def_path, lefs, rl, top):
    t = Path(def_path).read_text(errors='replace')
    used = set(re.findall(r'^\s*-\s+\S+\s+(\S+)', section(t, 'COMPONENTS'), re.M))
    k = rl.index(top) + 1 if top in rl else len(rl)
    ok, above = set(rl[:k]), set(rl[k:])
    det, low, high = [], [], []
    for l in lefs:
        lt = Path(l).read_text(errors='replace')
        for m in re.finditer(r'^[ \t]*MACRO[ \t]+(\S+)[ \t]*\r?\n(.*?)^[ \t]*END[ \t]+\1\b', lt, re.M | re.S):
            if m.group(1) not in used:
                continue
            row = []
            for pm in re.finditer(r'^[ \t]*PIN[ \t]+(\S+)[ \t]*\r?\n(.*?)^[ \t]*END[ \t]+\1\b', m.group(2), re.M | re.S):
                lays = sorted(set(re.findall(r'^\s*LAYER\s+(\S+)', pm.group(2), re.M)))
                use = val(pm.group(2), 'USE') or 'SIGNAL'
                row.append(f'{pm.group(1)}[{use[:3]}]:{"/".join(lays)}')
                if use in ('POWER', 'GROUND') or any(x in ok for x in lays):
                    continue
                (high if any(x in above for x in lays) else low).append(f'{m.group(1)}/{pm.group(1)} ({"/".join(lays)})')
            det.append(f'{m.group(1)}: ' + '  '.join(row))
    issues = []
    if low:
        issues.append(f'{len(low)} signal pin(s) only on device layers (GRT-0029; flow adds a {rl[0]} pad): ' + ', '.join(low[:10]))
    if high:
        issues.append(f'{len(high)} signal pin(s) only above {top} (GRT-0029; flow adds a {top} pad): ' + ', '.join(high[:10]))
    return dict(status='WARN' if issues else 'OK', details=det + (['--- issues (repaired by the flow) ---'] + issues if issues else []),
                key=issues, summary=f'{len(low)} pin(s) only on device layers, {len(high)} only above {top}')


@check('6 DEF raw OpenROAD read (original LEF + DEF)')
def c_def_raw(lefs, def_path):
    errs = P.check_def(lefs, def_path)
    return dict(status='WARN' if errs else 'OK', details=distinct(errs or []), key=distinct(errs or [], 6),
                summary=f'{len(errs)} error line(s) (the flow repairs a copy, see check 7)' if errs else 'reads as given')


@check('7 flow setup: all repairs (LEF, DEF, pins, router preflight)')
def c_setup(def_path, lefs, layers):
    case = C.own_case(Path(def_path).stem, def_path, lefs, layers, None, 20)
    return dict(summary=f'OK: layers {case["layers"]}, coupling {"proxy (um)" if case["rules"] == "proxy" else "OpenRCX"}',
                case=case)


@check('8 DEF write / read round trip (OpenROAD writes, then reads its own file)')
def c_roundtrip(case, d):
    out = d / 'roundtrip.def'
    C.run_openroad(C.OR_DIR / 'extract.tcl', dict(LEFS=' '.join(case['lefs']), DEF_IN=case['def'], RULES='proxy',
                   SPEF_OUT=d / 'roundtrip.spef', DEF_OUT=out), d / 'roundtrip_write.log')
    errs = P.check_def(case['lefs'], out)
    return dict(status='FAIL' if errs else 'OK', details=distinct(errs or []), key=distinct(errs or [], 6),
                summary=f'{len(errs)} error(s) reading back the written DEF' if errs else 'written DEF reads back')


@check('9 geometry export (what the coupling proxy measures)')
def c_export(case, d):
    out = d / 'geo.json'
    C.run_openroad(C.OR_DIR / 'export_geo.py', {}, d / 'export.log', python=True,
                   args=[','.join(case['lefs']), case['def'], out])
    g = json.loads(out.read_text())
    sig = {n: v for n, v in g['nets'].items() if v['signal_type'] in C.SIG}
    routed = [n for n, v in sig.items() if v['routed']]
    shapes = sum(len(v['shapes']) for v in sig.values())
    return dict(status='OK' if routed else 'FAIL', summary=f'{len(sig)} signal nets, {len(routed)} routed, {shapes} wire shapes',
                geo=g)


@check('10 reroute test (router default, one net at a time)')
def c_route(case, d, n_nets, geo):
    sig = [n for n, v in geo['nets'].items() if v['signal_type'] in C.SIG and v['routed'] and len(v['pins']) >= 2
           and n.replace(chr(92), '') not in set(case.get('fixed_nets', []))]
    if not sig:
        return dict(status='FAIL', summary='no routed signal net with >= 2 pins to test')
    by = sorted(sig, key=lambda n: (len(geo['nets'][n]['pins']), n))
    idx = sorted({round(i * (len(by) - 1) / max(1, n_nets - 1)) for i in range(n_nets)}) if n_nets > 1 else [0]
    picks = [by[i] for i in idx]   # smallest, middle ... largest net
    det, ok, logs = [], 0, []
    for j, n in enumerate(picks):
        rd = d / f'route_{j:02d}'; rd.mkdir(parents=True, exist_ok=True)
        log = rd / 'route.log'; logs.append(log)
        t0 = time.time()
        try:
            C.run_openroad(C.OR_DIR / 'route.tcl', dict(LEFS=' '.join(case['lefs']), STATE_DEF=case['def'], OUT_DEF=rd / 'out.def',
                           LAYERS=case['layers'], TARGETS='{' + n + '}', MODE='default', GUIDE=rd / 'grt.guide',
                           DRC=rd / 'drc.rpt'), log, timeout=3600)
            drc = len(re.findall(r'violation type', (rd / 'drc.rpt').read_text(errors='replace'))) if (rd / 'drc.rpt').exists() else 0
            det.append(f'OK   {n} ({len(geo["nets"][n]["pins"])} pins) {time.time() - t0:.0f}s, DRC markers {drc}'); ok += 1
        except Exception:   # noqa: BLE001
            det.append(f'FAIL {n} ({len(geo["nets"][n]["pins"])} pins) {time.time() - t0:.0f}s, log {log}')
    errs = []
    for lg in logs:
        if lg.exists():
            errs += [l for l in lg.read_text(errors='replace').splitlines()
                     if re.search(r'\[ERROR|^ERROR|^Error', l.strip()) and 'GRT-0253' not in l]
    if errs:
        det += ['--- router error lines ---'] + distinct(errs)
    return dict(status='OK' if ok == len(picks) else 'FAIL' if ok == 0 else 'WARN', details=det,
                key=det[:len(picks)] + distinct(errs, 8),
                summary=f'{ok}/{len(picks)} reroutes succeeded' + (f'; first error: {errs[0].strip()[:160]}' if errs and ok < len(picks) else ''))


# ------------------------------------------------------------------------------------------------ report
def write_report(d):
    """full detail in report.txt; the terminal gets the key lines of every check that is not OK"""
    nl = chr(10)
    (d / 'report.txt').write_text((nl + nl).join(f'[{c["status"]}] {c["name"]}: {c["summary"]}' + nl + nl.join(c['details'])
                                                for c in CHECKS) + nl)
    print(nl + '=' * 30 + ' key lines ' + '=' * 30)
    for c in CHECKS:
        if c['status'] == 'OK' and not c['name'].startswith(('7', '10')):
            continue
        print(f'[{c["status"]}] {c["name"]}')
        for l in (c.get('key') or [])[:12]:
            print('   ' + l.strip()[:170])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('def_file'); ap.add_argument('lefs', nargs='+')
    ap.add_argument('--layers'); ap.add_argument('--nets', type=int, default=3, help='nets for the reroute test (default 3)')
    ap.add_argument('--out', help='report folder (default out/diagnose/<DEF name>)')
    a = ap.parse_args()
    d = Path(a.out or REPO / 'out' / 'diagnose' / Path(a.def_file).stem).resolve(); d.mkdir(parents=True, exist_ok=True)
    print(f'diagnosis of {a.def_file} (all checks run; full detail in {d / "report.txt"})')
    c_env()
    lefs = [str(x) for x in C.lef_files(a.lefs)]
    c_lef_raw(lefs)
    rl = C.lef_routing_layers(lefs)
    top = a.layers.split('-')[-1] if a.layers else ''
    _, setup = c_setup(a.def_file, a.lefs, a.layers)
    case = setup.get('case') if setup else None
    if case:
        top = case['layers'].split('-')[-1]
    c_tech(lefs[0], top or (rl[-1] if rl else ''))
    c_def(a.def_file, lefs, top)
    c_pins(a.def_file, lefs, rl, top or (rl[-1] if rl else ''))
    c_def_raw(lefs, a.def_file)
    if case:
        c_roundtrip(case, d)
        _, ex = c_export(case, d)
        if ex:
            c_route(case, d, a.nets, ex['geo'])
    else:
        CHECKS.append(dict(name='8-10 round trip / export / reroute', status='FAIL', seconds=0.,
                           summary='skipped: the flow setup (check 7) failed', details=[]))
    CHECKS.sort(key=lambda c: int(re.match(r'\d+', c['name']).group()))
    write_report(d)
    bad = [c for c in CHECKS if c['status'] == 'FAIL']
    print(f'\n{len(CHECKS)} checks: {sum(c["status"] == "OK" for c in CHECKS)} OK, '
          f'{sum(c["status"] == "WARN" for c in CHECKS)} WARN (repaired), {len(bad)} FAIL')
    print(f'full detail: {d / "report.txt"}')


if __name__ == '__main__':
    main()
