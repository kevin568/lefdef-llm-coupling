#!/usr/bin/env python3
"""LEF/DEF -> candidate reroutes -> LLM ranking -> OpenROAD ECO route -> OpenRCX extraction -> coupling report.

  python flow/run.py tempsense                      # all stages, LLM policy, route the model's best tree per net
  python flow/run.py tempsense --policy heuristic   # no GPU needed
  python flow/run.py tempsense --route all          # route every candidate (ground truth for ranking metrics)
  python flow/run.py tempsense --from s4_route      # re-run from one stage (earlier stages must be ok)
  python flow/run.py A.def B.lef                    # your own layout: routed DEF + LEF files or folders

Every stage writes out/<case>/<stage>/status.json and keeps every OpenROAD log next to its outputs.
s6_apply commits the accepted reroutes one after another into one layout: out/<case>/s6_apply/final.def.
"""
import argparse
import json
import re
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import common as C                     # noqa: E402
from lib import spef, guides as G, candidates as K, features as F, proxy, proxyspef  # noqa: E402

STAGES = ['s0_check', 's1_import', 's2_candidates', 's3_score', 's4_route', 's5_extract', 's6_apply', 's7_report']


CU = 'fF'   # coupling unit shown in reports: 'um (proxy)' when no OpenRCX rules file is used


def pct(x):
    return f'{100. * (x["cc_after_fF"] - x["cc_before_fF"]) / x["cc_before_fF"]:+.1f}%' if x['cc_before_fF'] else '—'


def safe(name):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('_')[:40]


def def_nets(path):
    t = Path(path).read_text(errors='replace'); s = t[t.index('\nNETS'):t.index('END NETS')]
    out = {}
    for blk in s.split(';')[:-1]:
        blk = blk.strip(); m = re.match(r'-\s+(\S+)', blk)
        if m: out[m.group(1)] = ' '.join(blk.split())
    return out


def net_counts(ref):
    """counts only (no names): why no net qualifies"""
    conn = lambda t: len(re.findall(r'\(\s*\S+\s+\S+\s*\)', t.split('+')[0]))
    pg = [n for n, t in ref.items() if re.search(r'USE\s+(POWER|GROUND)', t)]
    wired = [n for n, t in ref.items() if n not in pg and re.search(r'\+\s*(ROUTED|FIXED|COVER)\s', t)]
    h = {'0-1': 0, '2-8': 0, '>8': 0}
    for n in wired:
        c = conn(ref[n]); h['0-1' if c < 2 else '2-8' if c <= 8 else '>8'] += 1
    return (f'NETS: {len(ref)} nets, {len(pg)} power/ground, {len(ref) - len(pg) - len(wired)} signal nets without wiring, '
            f'{len(wired)} with wiring (connections per wired net: 0-1: {h["0-1"]}, 2-8: {h["2-8"]}, >8: {h[">8"]})')


def tcl_list(nets):
    return ' '.join('{' + n + '}' for n in nets)


def route_job(case, state_def, nets, mode, rundir, guide_rects=None):
    rundir = Path(rundir); rundir.mkdir(parents=True, exist_ok=True)
    guide = rundir / ('targets.guide' if mode == 'guide' else 'grt.guide')
    if mode == 'guide':
        guide.write_text(''.join(f'{n}\n(\n' + ''.join(f'{a} {b} {c} {d} {l}\n' for a, b, c, d, l in guide_rects[n]) + ')\n' for n in nets))
    sec = C.run_openroad(C.OR_DIR / 'route.tcl', dict(LEFS=' '.join(case['lefs']), STATE_DEF=state_def, OUT_DEF=rundir / 'out.def',
                         LAYERS=case['layers'], TARGETS=tcl_list(nets), MODE=mode, GUIDE=guide, DRC=rundir / 'drc.rpt'),
                         rundir / 'route.log', threads=ARGS.threads)
    return sec


def route_errors(logs):
    """the distinct error lines of these OpenROAD logs, with how many logs show each"""
    seen = {}
    for lg in logs:
        try:
            lines = Path(lg).read_text(errors='replace').splitlines()
        except OSError:
            continue
        for l in sorted({re.sub(r'(?<![-\w.])\d+(\.\d+)?\b', '#', x.strip()) for x in lines
                         if re.search(r'\[ERROR|^ERROR|^Error|FLOW_ERROR', x.strip())}):
            seen[l] = seen.get(l, 0) + 1
    if not seen:
        return '  (no error line in the logs: see the last lines of the first log)'
    return '  router errors (count of logs):\n' + '\n'.join(f'    {n:3d}x {l[:200]}' for l, n in
                                                         sorted(seen.items(), key=lambda kv: -kv[1])[:15])


def try_route(case, state_def, net, rundir):
    """Router-default reroute of one net; False (instead of stopping the flow) when OpenROAD fails on it."""
    try:
        route_job(case, state_def, [net], 'default', rundir); return True
    except C.StageError:
        return False


def extract_job(case, def_in, spef_out, log, def_out=''):
    """OpenRCX extraction into spef_out; with rules 'proxy' (no rules file) the geometric coupling proxy is written
    as the SPEF instead (lib/proxyspef.py), so the later stages work unchanged."""
    sec = C.run_openroad(C.OR_DIR / 'extract.tcl', dict(LEFS=' '.join(case['lefs']), DEF_IN=def_in, RULES=case['rules'],
                         SPEF_OUT=spef_out, DEF_OUT=def_out), log, threads=ARGS.threads)
    if case['rules'] == 'proxy':
        g = Path(str(spef_out) + '.geo.json')
        export_job(case, def_in, g, Path(str(log) + '.geo.log'))
        proxyspef.write_spef(json.loads(g.read_text()), spef_out)
    return sec


def export_job(case, def_in, out_json, log):
    return C.run_openroad(C.OR_DIR / 'export_geo.py', {}, log, python=True, args=[','.join(case['lefs']), def_in, out_json])


def design_metrics(geo, sp):
    sig = [v for v in geo['nets'].values() if v['signal_type'] in C.SIG]
    return dict(cc_fF=sp['total_cc'], wl_um=sum(v['wl_dbu'] for v in sig) / geo['dbu'], vias=sum(v['vias'] for v in sig))


def pm_nets(geo):
    return {n: [(2 * geo['layers'][s['layer']]['level'], 0, s['rect']) for s in v['shapes']]
            for n, v in geo['nets'].items() if v['signal_type'] in C.SIG}


def fixed_nets(case):
    return {x.replace(chr(92), '') for x in case.get('fixed_nets', [])}


def net_ok(case, geo, n):
    v = geo['nets'][n]
    if v['signal_type'] not in C.SIG or not v['routed'] or not (2 <= len(v['pins']) <= case.get('max_pins', 8)):
        return False
    if n.replace(chr(92), '') in fixed_nets(case):   # touches a cell without a LEF: kept as it is
        return False
    if case.get('net_regex') and not re.search(case['net_regex'], n):
        return False
    if case.get('net_filter') == 'vco_macro_pin':   # analog VCO nets: at least one pin on a VCO macro instance
        return any(p.get('macro') and p['name'].split('/')[0].startswith('vco_') and not p['name'].split('/')[0].startswith('vco_adc')
                   for p in v['pins'])
    return True


# ------------------------------------------------------------------ stages
def s0_check(case, out, st):
    log = st.dir / 'openroad_check.log'
    probe = st.dir / 'probe.tcl'
    probe.write_text('foreach c {read_lef read_def read_guides global_route detailed_route extract_parasitics write_spef} '
                     '{ if {[llength [info commands $c]] == 0} { puts "MISSING $c" } }\nputs [ord::openroad_version]\nputs FLOW_DONE\n')
    C.run_openroad(probe, {}, log)
    txt = log.read_text()
    missing = re.findall(r'MISSING (\S+)', txt)
    if missing:
        raise C.StageError(f'OpenROAD lacks commands {missing}; log: {log}')
    st.info['openroad_version'] = [l for l in txt.splitlines() if l.strip() and 'FLOW_DONE' not in l][-1]
    # bundled inputs must be byte-identical to data/SOURCES.json (catches CRLF conversion or truncated downloads)
    src = {(C.REPO / s['path']).resolve(): s['sha256'] for s in json.loads((C.REPO / 'data/SOURCES.json').read_text())}
    for p in [case['def'], *([] if case['rules'] == 'proxy' else [case['rules']]), *case['lefs']]:
        want = src.get(Path(p).resolve())
        if want and C.sha256(p) != want:
            raise C.StageError(f'input file differs from data/SOURCES.json: {p}\n  (line-ending conversion on checkout? see README "Debugging")')
    st.info['inputs_verified'] = True
    st.info['cell_types_used'] = C.check_lef_def(case)
    if ARGS.policy == 'llm':
        try:
            import torch, transformers, peft, bitsandbytes  # noqa: F401
        except Exception as e:
            raise C.StageError(f'LLM policy needs torch/transformers/peft/bitsandbytes ({e}); pip install -r requirements.txt or use --policy heuristic')
        if not torch.cuda.is_available():
            raise C.StageError('LLM policy needs a CUDA GPU; use --policy heuristic to run without it')
        st.info['gpu'] = torch.cuda.get_device_name()
    st.info['case'] = {k: case[k] for k in ('def', 'layers', 'rules', 'split', 'domain', 'normalize', 'tolerate_baseline_drc') if k in case}


def known_violations(case, out):
    """Violations not counted as new: the case's known_violations, plus (with tolerate_baseline_drc) the pre-existing ones
    found in s2_candidates/baseline_violations.json."""
    k = [tuple(v) for v in case.get('known_violations', [])]
    f = out / 's2_candidates/baseline_violations.json'
    if case.get('tolerate_baseline_drc') and f.exists():
        k += [tuple(v) for v in json.loads(f.read_text())['violations']]
    return k


def auto_normalize(case, d, base_def, st):
    """Some published DEFs are re-patched by OpenROAD's detailed router (a reroute then also changes other nets near
    the rerouted one). Check: reroute, one at a time and with the router default (exactly what the later stages do), the
    likely target nets (the pool_nets highest-coupling candidates of the input DEF, or the --nets matches) plus one short
    2-pin net elsewhere, and compare every other net. If nothing else changed, the DEF is used as it is. Otherwise the
    re-patched nets are rerouted (router default) into a new baseline and the check is repeated (at most 4 checks)."""
    ref = def_nets(base_def)
    conns = {n: len(re.findall(r'\(\s*\S+\s+\S+\s*\)', t.split('+')[0])) for n, t in ref.items()}
    sig = [n for n, t in ref.items() if 'ROUTED' in t and not re.search(r'USE\s+(POWER|GROUND)', t)]
    if not sig:   # layouts exported with FIXED / COVER wiring only
        sig = [n for n, t in ref.items() if re.search(r'\+\s*(FIXED|COVER)\s', t) and not re.search(r'USE\s+(POWER|GROUND)', t)]
    rx = case.get('net_regex')
    cc = spef.read(d / 'input.spef')['listed_cc']
    fx = fixed_nets(case)
    sig = [n for n in sig if n.replace(chr(92), '') not in fx]
    mine = [n for n in sig if not rx or re.search(rx, n)]
    if not any(2 <= conns[n] <= case.get('max_pins', 8) for n in mine) and any(conns[n] > 8 for n in mine):
        case['max_pins'] = 10 ** 6   # analog layouts (multi-finger devices): every net has > 8 connections
    pool = sorted((n for n in sig if 2 <= conns[n] <= case.get('max_pins', 8) and (not rx or re.search(rx, n))),
                  key=lambda n: (-cc.get(n, 0.), n))[:case.get('pool_nets', 20)]
    two_pin = [n for n in sig if conns[n] == 2 and n not in pool and not (rx and re.search(rx, n))]
    probe = min(two_pin, key=lambda n: (len(ref[n]), n)) if two_pin else None
    check = ([probe] if probe else []) + pool
    if not check:
        info = dict(mode='auto', skipped='no routed signal net to check with', patched=False, stable=None)
        (d / 'normalize.json').write_text(json.dumps(info, indent=1))
        print('  note: re-patch check skipped (no routed signal net to check with); the DEF is used as it is. ' + net_counts(ref))
        return base_def
    cur, rounds, dirs = base_def, [], {}
    for k in range(4):
        cd = d / f'normalize_check_{k + 1}'
        dirs = {n: cd / f'{j:02d}_{safe(n)}' for j, n in enumerate(check)}
        with ThreadPoolExecutor(ARGS.jobs) as ex:
            ok = dict(zip(check, ex.map(lambda n: try_route(case, cur, n, dirs[n]), check)))
        failed = [n for n in check if not ok[n]]
        dirs = {n: p for n, p in dirs.items() if ok[n]}
        old = def_nets(cur); changed = set(); per_net = {}
        for n in dirs:
            new = def_nets(dirs[n] / 'out.def')
            ch = sorted(x for x in old if x != n and old[x] != new.get(x))
            per_net[n] = len(ch); changed |= set(ch)
        changed = sorted(changed)
        rounds.append(dict(kind='check', base=str(cur), nets=len(check), other_nets_changed=len(changed),
                           examples=changed[:10], per_net=per_net, route_failed=failed))
        if not changed or k == 3:
            break
        pd = d / f'normalize_patch_{k + 1}'   # reroute the re-patched nets (router default) into a new baseline
        try:
            route_job(case, cur, changed, 'default', pd)
        except C.StageError:
            rounds.append(dict(kind='patch', nets=len(changed), route_failed=True, other_nets_changed=len(changed)))
            break   # keep the last good baseline; reported as not stable
        rounds.append(dict(kind='patch', nets=len(changed), drc_violations=len(G.violations(pd / 'drc.rpt'))))
        cur = pd / 'out.def'
    stable = rounds[-1]['other_nets_changed'] == 0 and bool(dirs)   # no check reroute succeeded: nothing verified
    info = dict(mode='auto', probe=probe, check_nets=check, patched=cur != base_def, stable=stable, rounds=rounds,
                drc=str(next(iter(dirs.values())) / 'drc.rpt') if stable and dirs else None,   # an unstable run's violations are not tolerated
                route_dirs={n: str(p) for n, p in dirs.items()} if stable else {})   # reused by s2_candidates
    (d / 'normalize.json').write_text(json.dumps(info, indent=1))
    st.info.update(def_repatched=info['patched'], normalize_stable=stable)
    if info['patched'] or not stable:
        first = rounds[0]
        print(f'  note: OpenROAD re-patches this DEF (rerouting {len(check)} nets one at a time changed '
              f'{first["other_nets_changed"]} other nets); '
              + (f'baseline = DEF with those nets rerouted by the router default ({sum(r["kind"] == "patch" for r in rounds)} '
                 'patch round(s))' if stable else
                 f'still not stable after {sum(r["kind"] == "patch" for r in rounds)} patch rounds '
                 f'({rounds[-1]["other_nets_changed"]} nets keep changing): expect few valid reroutes '
                 '(the DEF may not match these LEFs)'))
    return cur


def s1_import(case, out, st):
    d = st.dir
    extract_job(case, case['def'], d / 'input.spef', d / 'import.log', def_out=d / 'input_rewritten.def')
    base_def = d / 'input_rewritten.def'
    norm = case.get('normalize')
    if norm == 'auto':
        base_def = auto_normalize(case, d, base_def, st)
    elif norm:   # deterministic DRT patching of pre-existing defects in the public DEF (see README, "vco_macro")
        cur = base_def
        for i in range(norm['times']):
            rd = d / f'normalize_{i + 1}'
            route_job(case, cur, [norm['net']], 'default', rd); cur = rd / 'out.def'
        base_def = cur
        (d / 'normalize.json').write_text(json.dumps(dict(mode='net', net=norm['net'], drc=str(rd / 'drc.rpt')), indent=1))
    (d / 'base.def').write_bytes(Path(base_def).read_bytes())
    extract_job(case, d / 'base.def', d / 'base.spef', d / 'extract.log')
    export_job(case, d / 'base.def', d / 'geo.json', d / 'export.log')
    geo = json.loads((d / 'geo.json').read_text()); sp = spef.read(d / 'base.spef')
    m = design_metrics(geo, sp)
    m.update(instances=geo['instances'], signal_nets=sum(v['signal_type'] in C.SIG for v in geo['nets'].values()),
             coupling_caps=sp['n_coupling_caps'], base_def_sha256=C.sha256(d / 'base.def'))
    (d / 'baseline.json').write_text(json.dumps(m, indent=1))
    st.info.update(m)


def s2_candidates(case, out, st):
    d = st.dir; b = out / 's1_import'
    geo = json.loads((b / 'geo.json').read_text()); sp = spef.read(b / 'base.spef')
    base_def = b / 'base.def'
    eligible = [n for n in geo['nets'] if net_ok(case, geo, n)]
    if not eligible and 'max_pins' not in case:   # analog layouts: every net has > 8 connections; take them all
        case['max_pins'] = 10 ** 6
        eligible = [n for n in geo['nets'] if net_ok(case, geo, n)]
        if eligible:
            print(f'  note: no net has 2..8 connections (multi-finger devices); all {len(eligible)} routed signal nets '
                  'with >= 2 connections are candidates')
    if not eligible:
        raise C.StageError('no eligible net (signal, routed, 2..8 pins, case filter'
                           + (f'; {len(case["fixed_nets"])} nets touch cells without a LEF and are kept as they are: '
                              'give the LEF that defines those cells' if case.get('fixed_nets') else '') + '). '
                           + net_counts(def_nets(base_def)))
    nets = sorted(eligible, key=lambda n: (-sp['listed_cc'].get(n, 0.), n))[:case.get('pool_nets', 20)]
    # pin access: OpenROAD's own reroute of each pool net (its global-route guide gives the pin-access gcells;
    # the same run is also the "default" (router's choice) candidate of that net)
    pdirs = {n: d / 'pin_access' / f'{j:02d}_{safe(n)}' for j, n in enumerate(nets)}
    nj = out / 's1_import/normalize.json'
    plain = lambda s: s.replace('\\', '')   # DEF names may escape bus brackets, OpenDB names do not
    done = {plain(k): Path(v) for k, v in
            (json.loads(nj.read_text()).get('route_dirs') or {}).items()} if nj.exists() else {}
    for n in nets:   # the auto-normalize check already rerouted these nets, one at a time, on this same baseline
        if plain(n) in done and (done[plain(n)] / 'out.def').exists():
            pdirs[n] = done[plain(n)]
    todo = [n for n in nets if pdirs[n].parent.name == 'pin_access']
    with ThreadPoolExecutor(ARGS.jobs) as ex:
        ok = dict(zip(todo, ex.map(lambda n: try_route(case, base_def, n, pdirs[n]), todo)))
    bad = [n for n in todo if not ok[n]]
    if bad:   # OpenROAD could not reroute these nets at all: leave them out of the pool
        print(f'  note: {len(bad)} net(s) left out: the router-default reroute failed (e.g. {bad[0]}, log: '
              f'{pdirs[bad[0]] / "route.log"})')
        nets = [n for n in nets if n not in bad]
        if not nets:
            raise C.StageError(f'every router-default reroute failed; first log: {pdirs[bad[0]] / "route.log"}\n'
                               + route_errors([pdirs[n] / 'route.log' for n in bad]))
    st.info['route_failed_nets'] = bad
    # pre-existing DRC violations: reported by every router-default reroute (>= 2 nets) or by the last normalize run;
    # they are tolerated for your own layouts unless --strict (tolerate_baseline_drc in cases.json)
    sets = [set(G.violations(pdirs[n] / 'drc.rpt')) for n in nets]
    pre = set.intersection(*sets) if len(sets) >= 2 else set()
    if nj.exists() and json.loads(nj.read_text()).get('drc'):
        pre |= set(G.violations(json.loads(nj.read_text())['drc']))
    (d / 'baseline_violations.json').write_text(json.dumps(dict(violations=sorted(pre)), indent=1))
    st.info.update(baseline_violations=len(pre), tolerated=bool(case.get('tolerate_baseline_drc')))
    if pre and case.get('tolerate_baseline_drc'):
        kinds = {}
        for v in pre:
            kinds[v[0]] = kinds.get(v[0], 0) + 1
        print(f'  note: {len(pre)} pre-existing DRC violation(s) are not counted as new ('
              + ', '.join(f'{n} {k}' for k, n in sorted(kinds.items())) + f'; list: {d / "baseline_violations.json"}; --strict counts them)')
    grid = G.gcell_grid(pdirs[nets[0]] / 'route.log')
    if not grid:
        raise C.StageError(f'could not read GCELLGRID from {pdirs[nets[0]] / "route.log"}')
    hl, vl, minlev = G.corridor_layers(geo, case['layers'])
    F.setup_units(geo); pmn = pm_nets(geo)
    dm = design_metrics(geo, sp)
    block = dict(C=proxy.coupling(pmn, set())[0], WL=dm['wl_um'], vias=dm['vias'])   # design context line of the prompt
    margin = 10 * geo['dbu']; cands = []
    for n in nets:
        pins = [p['xy'] for p in geo['nets'][n]['pins']]
        pb = G.grt_pin_boxes(pdirs[n] / 'grt.guide', n, geo, minlev)
        cands.append(dict(candidate_id=f'{n}:default', net=n, kind='default', route_dir=str(pdirs[n])))
        for cd in K.pool_for_net(n, pins):
            g = cd['imported']['geometry']
            win = G.window_nets(pmn, [[min(s[0], s[2]), min(s[1], s[3]), max(s[0], s[2]), max(s[1], s[3])] for s in g]
                                + [r for _, _, r in pmn.get(n, [])], margin)
            win[n] = pmn.get(n, [])
            f = F.analyse(win, set(), 0, cd, block)
            cd.update(guides=G.guides_for(g, pins, pb, grid, hl, vl), prompt=F.render(f, case['name'], case['name']),
                      heuristic=f['removed_cpl_um'] - f['new_same_layer_um'], feature=f)
            cands.append(cd)
    (d / 'candidates.json').write_text(json.dumps(dict(case=case['name'], nets=nets, grid=grid, corridor=[hl, vl],
                                                       proxy_block=block, candidates=cands)) + '\n')
    st.info.update(nets=len(nets), candidates=len(cands), eligible_nets=len(eligible), corridor=[hl, vl])


def s3_score(case, out, st):
    import numpy as np
    P = json.loads((out / 's2_candidates/candidates.json').read_text())
    trees = [c for c in P['candidates'] if c['kind'] != 'default']
    rnd = np.random.default_rng(121)
    rows = [dict(candidate_id=c['candidate_id'], net=c['net'], kind=c['kind'], heuristic=c['heuristic'], random=float(rnd.random()))
            for c in trees]
    if ARGS.policy == 'llm':
        from lib.scorer import Scorer
        adapter = Path(ARGS.adapter) if ARGS.adapter else C.REPO / 'models/llm_v2/adapter'
        sc = Scorer(adapter); st.info['model'] = sc.identity
        for r, s in zip(rows, sc.score([c['prompt'] for c in trees])):
            r['llm'] = s
    (st.dir / 'scores.json').write_text(json.dumps(dict(policy=ARGS.policy, rows=rows), indent=1))
    st.info['scored'] = len(rows)


def s4_route(case, out, st):
    P = json.loads((out / 's2_candidates/candidates.json').read_text())
    S = {r['candidate_id']: r for r in json.loads((out / 's3_score/scores.json').read_text())['rows']}
    key = 'llm' if ARGS.policy == 'llm' else ARGS.policy
    trees = [c for c in P['candidates'] if c['kind'] != 'default']
    if ARGS.route == 'all':
        pick = trees
    else:   # the policy's best tree per net
        best = {}
        for c in trees:
            s = S[c['candidate_id']][key]
            if c['net'] not in best or s > S[best[c['net']]['candidate_id']][key]:
                best[c['net']] = c
        pick = [best[n] for n in P['nets'] if n in best]
    base_def = out / 's1_import/base.def'
    jobs = [(i, c, st.dir / 'runs' / f'{i:03d}_{safe(c["candidate_id"])}') for i, c in enumerate(pick)]

    def one(job):
        i, c, rd = job
        if (rd / 'out.def').exists() and (rd / 'route.log').exists() and 'FLOW_DONE' in (rd / 'route.log').read_text(errors='replace'):
            return rd
        route_job(case, base_def, [c['net']], 'guide', rd, {c['net']: c['guides']}); return rd
    failures = []
    with ThreadPoolExecutor(ARGS.jobs) as ex:
        futs = [(j, ex.submit(one, j)) for j in jobs]
        for (i, c, rd), fu in futs:
            try:
                fu.result()
            except C.StageError as e:
                failures.append(dict(candidate_id=c['candidate_id'], error=str(e).splitlines()[0], log=str(rd / 'route.log')))
    routed = [dict(candidate_id=c['candidate_id'], net=c['net'], kind=c['kind'], dir=str(rd)) for i, c, rd in jobs
              if (rd / 'out.def').exists()]
    routed += [dict(candidate_id=c['candidate_id'], net=c['net'], kind='default', dir=c['route_dir'])
               for c in P['candidates'] if c['kind'] == 'default']
    (st.dir / 'routed.json').write_text(json.dumps(dict(routed=routed, failures=failures), indent=1))
    st.info.update(routed=len(routed), failures=len(failures))
    if failures and len(failures) == len(jobs):
        raise C.StageError(f'every route failed; first log: {failures[0]["log"]}')


def s5_extract(case, out, st):
    b = out / 's1_import'
    base = json.loads((b / 'baseline.json').read_text()); base_sp = spef.read(b / 'base.spef'); ref = def_nets(b / 'base.def')
    R = json.loads((out / 's4_route/routed.json').read_text())['routed']
    known = known_violations(case, out)

    def one(r):
        rd = Path(r['dir']); m = dict(r)
        try:
            if not (rd / 'out.spef').exists():
                extract_job(case, rd / 'out.def', rd / 'out.spef', rd / 'extract.log')
            if not (rd / 'geo.json').exists():
                export_job(case, rd / 'out.def', rd / 'geo.json', rd / 'export.log')
        except C.StageError as e:
            m.update(valid=False, error=str(e).splitlines()[0]); return m
        geo = json.loads((rd / 'geo.json').read_text()); sp = spef.read(rd / 'out.spef'); n = r['net']
        viol = G.violations(rd / 'drc.rpt'); new_viol = [v for v in viol if v not in known]
        new_nets = def_nets(rd / 'out.def')
        changed = sorted(x for x in ref if x != n and ref[x] != new_nets.get(x))
        m.update(design_metrics(geo, sp))
        m.update(target_routed=geo['nets'][n]['routed'], new_violations=len(new_viol), known_violations=len(viol) - len(new_viol),
                 other_nets_changed=len(changed), net_cc_before_fF=base_sp['per_net_cc'].get(n, 0.), net_cc_after_fF=sp['per_net_cc'].get(n, 0.),
                 design_dcc_fF=sp['total_cc'] - base['cc_fF'])
        m['valid'] = bool(m['target_routed'] and not new_viol and not changed)
        m['U'] = C.utility(m, base)
        (rd / 'metrics.json').write_text(json.dumps(m, indent=1))
        return m
    with ThreadPoolExecutor(ARGS.jobs) as ex:
        rows = list(ex.map(one, R))
    (st.dir / 'measurements.json').write_text(json.dumps(dict(baseline=base, rows=rows), indent=1))
    st.info.update(measured=len(rows), valid=sum(bool(r.get('valid')) for r in rows))


def s6_apply(case, out, st):
    """Commit reroutes into one layout: for each net take the policy's top-scored routed tree, keep only those whose
    single-net U > 0, then reroute them one after another on the evolving DEF (largest U first). A step is kept when it
    is legal against the current layout and improves U again; otherwise it is skipped. Output: final.def / final.spef."""
    d = st.dir; b = out / 's1_import'
    M = json.loads((out / 's5_extract/measurements.json').read_text()); base = M['baseline']
    S = {r['candidate_id']: r for r in json.loads((out / 's3_score/scores.json').read_text())['rows']}
    cand = {c['candidate_id']: c for c in json.loads((out / 's2_candidates/candidates.json').read_text())['candidates']}
    key = ARGS.policy
    pick = {}
    for r in M['rows']:
        if r['kind'] == 'default' or not r.get('valid'):
            continue
        if r['net'] not in pick or S[r['candidate_id']][key] > S[pick[r['net']]['candidate_id']][key]:
            pick[r['net']] = r
    queue = sorted((r for r in pick.values() if r['U'] > 0), key=lambda r: -r['U'])
    known = known_violations(case, out)
    base_sp = spef.read(b / 'base.spef')
    cur_def, cur_spef, cur_m, cur_sp, cur_nets = b / 'base.def', b / 'base.spef', dict(base), base_sp, def_nets(b / 'base.def')
    steps = []
    for i, r in enumerate(queue):
        n = r['net']; sd = d / 'steps' / f'{i:02d}_{safe(n)}'
        rec = dict(step=i, net=n, candidate_id=r['candidate_id'], single_U=r['U'], dir=str(sd))
        try:
            route_job(case, cur_def, [n], 'guide', sd, {n: cand[r['candidate_id']]['guides']})
            extract_job(case, sd / 'out.def', sd / 'out.spef', sd / 'extract.log')
            export_job(case, sd / 'out.def', sd / 'geo.json', sd / 'export.log')
        except C.StageError as e:
            rec.update(accepted=False, reason=str(e).splitlines()[0]); steps.append(rec); continue
        geo = json.loads((sd / 'geo.json').read_text()); sp = spef.read(sd / 'out.spef'); m = design_metrics(geo, sp)
        new_viol = [v for v in G.violations(sd / 'drc.rpt') if v not in known]
        new_nets = def_nets(sd / 'out.def')
        changed = sorted(x for x in cur_nets if x != n and cur_nets[x] != new_nets.get(x))
        step_u = C.utility(m, cur_m)
        why = [w for w, bad in [('target not routed', not geo['nets'][n]['routed']), (f'{len(new_viol)} new DRC violation(s)', new_viol),
                                (f'{len(changed)} other net(s) changed', changed), (f'no further gain (step U {step_u:+.4f})', step_u <= 0)] if bad]
        rec.update(accepted=not why, reason=', '.join(why), step_U=step_u, **m)
        if not why:
            cur_def, cur_spef, cur_m, cur_sp, cur_nets = sd / 'out.def', sd / 'out.spef', m, sp, new_nets
        steps.append(rec)
    (d / 'final.def').write_bytes(Path(cur_def).read_bytes()); (d / 'final.spef').write_bytes(Path(cur_spef).read_bytes())
    acc = [s for s in steps if s['accepted']]
    nets = [dict(net=s['net'], candidate_id=s['candidate_id'], cc_before_fF=base_sp['per_net_cc'].get(s['net'], 0.),
                 cc_after_fF=cur_sp['per_net_cc'].get(s['net'], 0.)) for s in acc]
    rel = lambda a, z: 100. * (a - z) / z if z else 0.
    summ = dict(policy=ARGS.policy, key_metrics=key_metrics(case, out, nets, cur_sp), baseline={k: base[k] for k in ('cc_fF', 'wl_um', 'vias')}, final=cur_m, U=C.utility(cur_m, base),
                cc_change_pct=rel(cur_m['cc_fF'], base['cc_fF']), wl_change_pct=rel(cur_m['wl_um'], base['wl_um']),
                via_change=cur_m['vias'] - base['vias'], tried=len(queue), accepted=len(acc), nets=nets,
                final_def=str(d / 'final.def'), final_spef=str(d / 'final.spef'), steps=steps)
    (d / 'apply.json').write_text(json.dumps(summ, indent=1))
    st.info.update(tried=len(queue), accepted=len(acc), cc_before_fF=round(base['cc_fF'], 3), cc_after_fF=round(cur_m['cc_fF'], 3),
                   cc_change_pct=round(summ['cc_change_pct'], 2))


def key_metrics(case, out, nets, final_sp=None):
    """Headline metrics: the rerouted nets, the target set (nets the flow examined: top coupling or --nets), worst net."""
    rel = lambda a, z: 100. * (a - z) / z if z else 0.
    base_sp = spef.read(out / 's1_import/base.spef'); final_sp = final_sp or spef.read(out / 's6_apply/final.spef')
    pool = json.loads((out / 's2_candidates/candidates.json').read_text())['nets']
    tb = {n: base_sp['per_net_cc'].get(n, 0.) for n in pool}; ta = {n: final_sp['per_net_cc'].get(n, 0.) for n in pool}
    rb, ra = sum(x['cc_before_fF'] for x in nets), sum(x['cc_after_fF'] for x in nets)
    per = sorted(rel(x['cc_after_fF'], x['cc_before_fF']) for x in nets if x['cc_before_fF'])
    wb = max(tb, key=tb.get) if tb else None; wa = max(ta, key=ta.get) if ta else None
    return dict(
        rerouted_nets=dict(n=len(nets), cc_before_fF=rb, cc_after_fF=ra, change_pct=rel(ra, rb),
                           median_net_change_pct=per[len(per) // 2] if per else None,
                           nets_freed=sum(x['cc_after_fF'] <= .01 * x['cc_before_fF'] for x in nets)),
        target_nets=dict(n=len(pool), cc_before_fF=sum(tb.values()), cc_after_fF=sum(ta.values()),
                         change_pct=rel(sum(ta.values()), sum(tb.values())), selection=case.get('net_regex') or f'top {len(pool)} by coupling'),
        worst_net=dict(before=[wb, tb.get(wb, 0.)], after=[wa, ta.get(wa, 0.)]))


def load_apply(case, out):
    ap = out / 's6_apply/apply.json'
    if not ap.exists():
        return None
    A = json.loads(ap.read_text())
    if 'key_metrics' not in A:   # outputs written before the headline metrics existed
        A['key_metrics'] = key_metrics(case, out, A['nets']); ap.write_text(json.dumps(A, indent=1))
    return A


def s7_report(case, out, st):
    from statistics import mean
    M = json.loads((out / 's5_extract/measurements.json').read_text()); base = M['baseline']; rows = M['rows']
    S = {r['candidate_id']: r for r in json.loads((out / 's3_score/scores.json').read_text())['rows']}
    pol = 'llm' if ARGS.policy == 'llm' else ARGS.policy
    trees = [r for r in rows if r['kind'] != 'default']; dflt = {r['net']: r for r in rows if r['kind'] == 'default'}
    lines = [f'# Coupling report: {case["name"]}', '',
             f'Baseline: total coupling {base["cc_fF"]:.3f} {CU}, wire length {base["wl_um"]:.1f} um, vias {base["vias"]}, '
             f'{base["instances"]} instances, {base["signal_nets"]} signal nets.', '']
    if CU != 'fF':
        lines += ['Coupling is the geometric proxy (no OpenRCX rules file for this technology): um that a net runs '
                  'parallel to another net on the same layer within 2 routing pitches. Reroutes, DRC and the other-net '
                  'checks are real (OpenROAD); for fF, extract the input DEF and s6_apply/final.def with your extractor '
                  'or pass --rules FILE.', '']
    A = load_apply(case, out)
    if A:
        f = A['final']; K = A['key_metrics']; r, t = K['rerouted_nets'], K['target_nets']
        wb, wa = K['worst_net']['before'], K['worst_net']['after']
        med = f'{r["median_net_change_pct"]:+.1f}%' if r['median_net_change_pct'] is not None else '—'
        lines += ['## Before / after (s6_apply: accepted reroutes combined in one layout)', '',
                  '| key metric | before | after | change |', '|---|---:|---:|---:|',
                  f'| **coupling of the {r["n"]} rerouted nets ({CU})** | {r["cc_before_fF"]:.3f} | {r["cc_after_fF"]:.3f} | **{r["change_pct"]:+.2f}%** |',
                  f'| coupling of the {t["n"]} target nets ({t["selection"]}) ({CU}) | {t["cc_before_fF"]:.3f} | {t["cc_after_fF"]:.3f} | {t["change_pct"]:+.2f}% |',
                  f'| worst target net ({CU}) | {wb[1]:.3f} `{wb[0]}` | {wa[1]:.3f} `{wa[0]}` | {100. * (wa[1] - wb[1]) / wb[1] if wb[1] else 0.:+.2f}% |',
                  f'| design total coupling ({CU}) | {base["cc_fF"]:.3f} | {f["cc_fF"]:.3f} | {A["cc_change_pct"]:+.2f}% |',
                  f'| wire length (um) | {base["wl_um"]:.1f} | {f["wl_um"]:.1f} | {A["wl_change_pct"]:+.2f}% |',
                  f'| vias | {base["vias"]} | {f["vias"]} | {A["via_change"]:+d} |', '',
                  f'Rerouted nets: median per-net change {med}, {r["nets_freed"]} of {r["n"]} nets lose ≥99% of their coupling. '
                  'Design total includes nets the flow does not touch (e.g. long chip-level IO lines), so it is a context row.', '',
                  f'U = {A["U"]:+.4f}. {A["accepted"]} of {A["tried"]} candidate reroutes accepted. Final layout: `{A["final_def"]}` '
                  f'(SPEF `{A["final_spef"]}`).', '']
        if A['nets']:
            lines += [f'| rerouted net | coupling before → after ({CU}) | change |', '|---|---|---:|']
            lines += [f'| {x["net"]} | {x["cc_before_fF"]:.2f} → {x["cc_after_fF"]:.2f} | {pct(x)} |' for x in A['nets']]
            lines += ['']
        sk = [s for s in A['steps'] if not s['accepted']]
        if sk:
            lines += ['Skipped while combining: ' + '; '.join(f'{s["net"]} ({s["reason"]}, `{s["dir"]}`)' for s in sk), '']
    lines += ['## Single-net reroutes (each measured alone against the baseline)', '',
              f'Policy `{ARGS.policy}`, route mode `{ARGS.route}`. One net rerouted per row (OpenROAD ECO), measured by '
              f'{"OpenRCX" if CU == "fF" else "the geometric coupling proxy"}.', '',
             f'| net | candidate | valid | net CC before → after ({CU}) | design ΔCC ({CU}) | U | router default: net CC after / U |',
             '|---|---|---|---|---:|---:|---|']
    shown = trees if ARGS.route != 'all' else sorted(trees, key=lambda r: -S[r['candidate_id']].get(pol, 0))
    for r in shown:
        d = dflt.get(r['net'], {})
        dtxt = f'{d.get("net_cc_after_fF", float("nan")):.2f} / {d.get("U", float("nan")):+.4f}' if d.get('valid') else 'invalid'
        if r.get('error'):
            lines.append(f'| {r["net"]} | {r["candidate_id"]} | ERROR | {r["error"]} | | | {dtxt} |'); continue
        lines.append(f'| {r["net"]} | {r["kind"]} | {"yes" if r["valid"] else "no"} | {r["net_cc_before_fF"]:.2f} → {r["net_cc_after_fF"]:.2f} '
                     f'| {r["design_dcc_fF"]:+.3f} | {r["U"]:+.4f} | {dtxt} |')
    v = [r for r in trees if r.get('valid')]
    summ = dict(case=case['name'], policy=ARGS.policy, route=ARGS.route, baseline=base, routed=len(trees), valid=len(v),
                improved=sum(r['U'] > 0 for r in v), mean_U=mean(r['U'] for r in v) if v else None,
                net_cc_before_fF=sum(r['net_cc_before_fF'] for r in v), net_cc_after_fF=sum(r['net_cc_after_fF'] for r in v))
    lines += ['', f'Valid {len(v)}/{len(trees)}, improved {summ["improved"]}, '
                  f'rerouted nets coupling {summ["net_cc_before_fF"]:.2f} → {summ["net_cc_after_fF"]:.2f} {CU}.']
    if ARGS.route == 'all' and len(v) >= 3:   # ranking quality against measured U
        from scipy.stats import spearmanr
        rk = {}
        for k in [x for x in ('llm', 'heuristic', 'random') if x in S[v[0]['candidate_id']]]:
            o = sorted(v, key=lambda r: -S[r['candidate_id']][k]); bestU = max(r['U'] for r in v)
            rk[k] = dict(spearman=float(spearmanr([S[r['candidate_id']][k] for r in v], [r['U'] for r in v]).statistic),
                         p_at3=sum(r['U'] > 0 for r in o[:3]) / 3, top1_U=o[0]['U'], best_U=bestU)
        summ['ranking'] = rk
        lines += ['', '| ranker | Spearman vs measured U | precision@3 | top-1 U | best U |', '|---|---:|---:|---:|---:|']
        lines += [f'| {k} | {x["spearman"]:.3f} | {x["p_at3"]:.2f} | {x["top1_U"]:+.4f} | {x["best_U"]:+.4f} |' for k, x in rk.items()]
    bad = [r for r in rows if not r.get('valid')]
    if bad:
        lines += ['', '## Not valid (debug)', '', '| candidate | reason | run dir |', '|---|---|---|']
        for r in bad:
            why = r.get('error') or ', '.join(x for x, ok in [('target not routed', not r.get('target_routed', True)),
                                                               (f'{r.get("new_violations")} new DRC violation(s)', r.get('new_violations')),
                                                               (f'{r.get("other_nets_changed")} other net(s) changed', r.get('other_nets_changed'))] if ok)
            lines.append(f'| {r["candidate_id"]} | {why} | {r["dir"]} |')
    (out / 'report.md').write_text('\n'.join(lines) + '\n'); (out / 'report.json').write_text(json.dumps(summ, indent=1))
    st.info.update(valid=len(v), improved=summ['improved'])


def default_policy():
    try:
        import torch, transformers, peft, bitsandbytes  # noqa: F401
        if torch.cuda.is_available():
            return 'llm'
    except Exception:
        pass
    print('no CUDA GPU / LLM packages: using --policy heuristic (geometric ranking)')
    return 'heuristic'


def main():
    global ARGS, CU
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', metavar='A.def B.lef',
                    help='your layout: the routed DEF, then its LEF files or folders (technology LEF found automatically); '
                         'or a case name from config/cases.json')
    own = ap.add_argument_group('options for your own layout (all optional)')
    own.add_argument('--nets', help='regular expression: only nets whose name matches are rerouted, e.g. "vco|clk"')
    own.add_argument('--layers', help='signal routing layers, e.g. met1-met5 (default: every LEF routing layer except the lowest)')
    own.add_argument('--rules', help='OpenRCX rules file (default: bundled SKY130 rules for SKY130 layers; other technologies: '
                     '"proxy", the geometric coupling proxy in um)')
    own.add_argument('--pool-nets', type=int, default=20, help='how many highest-coupling nets to try (default 20)')
    own.add_argument('--normalize', default='auto', metavar='auto|off|NET[:TIMES]',
                     help='published DEFs that OpenROAD re-patches on its first run are detected and patched first (default auto)')
    own.add_argument('--strict', action='store_true',
                     help='count pre-existing DRC violations (reported by every router-default reroute) as new')
    own.add_argument('--def', dest='def_file', help=argparse.SUPPRESS)
    own.add_argument('--lef', help=argparse.SUPPRESS)
    ap.add_argument('--policy', default=None, choices=['llm', 'heuristic', 'random'],
                    help='default: llm when a CUDA GPU and the LLM packages are available, else heuristic')
    ap.add_argument('--adapter', help='LoRA adapter dir (default models/llm_v2/adapter)')
    ap.add_argument('--route', default='best', choices=['best', 'all'], help='best: policy top-1 per net; all: every candidate')
    ap.add_argument('--from', dest='start', choices=STAGES, help='re-run from this stage')
    ap.add_argument('--only', choices=STAGES, help='run a single stage')
    ap.add_argument('--out', default=None, help='output dir (default out/<DEF name or case>)')
    ap.add_argument('--jobs', type=int, default=3); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--cases', default=None, help='alternative cases.json')
    ARGS = ap.parse_args()
    try:
        first = ARGS.inputs[0]
        if ARGS.def_file or first.lower().endswith(('.def', '.def.gz')):
            def_file = ARGS.def_file or first
            lefs = ARGS.inputs[1:] + ([ARGS.lef] if ARGS.lef else [])
            ARGS.case = first if ARGS.def_file else re.sub(r'\.def(\.gz)?$', '', Path(first).name, flags=re.I)
            case = C.own_case(ARGS.case, def_file, lefs, ARGS.layers, ARGS.rules, ARGS.pool_nets, ARGS.normalize, ARGS.strict)
        else:
            if len(ARGS.inputs) > 1:
                raise C.StageError(f'"{first}" is not a .def file; give the DEF first: python flow/run.py A.def B.lef')
            ARGS.case = first
            case = C.load_case(ARGS.case, ARGS.cases)
        if ARGS.policy is None:
            ARGS.policy = default_policy()
        if ARGS.nets:
            case['net_regex'] = ARGS.nets
    except C.StageError as e:
        print(f'ERROR: {e}'); sys.exit(2)
    CU = 'um (proxy)' if case['rules'] == 'proxy' else 'fF'
    out = Path(ARGS.out or C.REPO / 'out' / ARGS.case).resolve(); out.mkdir(parents=True, exist_ok=True)
    # an output folder belongs to one set of inputs: finished stages are reused only for the same inputs
    fp = dict(def_sha256=C.sha256(case['def']), lef_sha256=[C.sha256(l) for l in case['lefs']], layers=case['layers'],
              rules_sha256='proxy' if case['rules'] == 'proxy' else C.sha256(case['rules']), nets=case.get('net_regex'), pool_nets=case.get('pool_nets'),
              normalize=case.get('normalize'), tolerate_baseline_drc=bool(case.get('tolerate_baseline_drc')))
    fpf = out / 'inputs.json'
    if fpf.exists() and json.loads(fpf.read_text()) != fp:
        old = json.loads(fpf.read_text())
        diff = ', '.join(k for k in fp if old.get(k) != fp[k]) or 'inputs'
        print(f'ERROR: the output folder {out} holds an earlier run with different {diff}.\n'
              f'  Start again in that folder (deletes the earlier results):  rm -rf {out}\n'
              f'  or keep them and write elsewhere: add  --out {out}_2  to the command'); sys.exit(2)
    fpf.write_text(json.dumps(fp, indent=1))
    todo = [ARGS.only] if ARGS.only else STAGES[STAGES.index(ARGS.start):] if ARGS.start else STAGES
    for name in STAGES:
        if name not in todo:
            if STAGES.index(name) < STAGES.index(todo[0]) and not C.stage_ok(out, name) and name != 's0_check':
                print(f'ERROR: stage {name} has not completed; run it first (status: {out / name / "status.json"})'); sys.exit(2)
            continue
        if not ARGS.start and not ARGS.only and C.stage_ok(out, name) and name not in ('s0_check',):
            print(f'[skip] {name} (already ok)'); continue
        print(f'[run ] {name} ...', flush=True)
        try:
            with C.Stage(out, name) as st:
                globals()[name](case, out, st)
            print(f'[ ok ] {name}  {json.dumps({k: v for k, v in st.info.items() if not isinstance(v, (dict, list))})}')
        except C.StageError as e:
            print(f'\n[FAIL] {name}\n{e}\n  status: {out / name / "status.json"}'); sys.exit(1)
        except Exception:
            print(f'\n[FAIL] {name} (unexpected error)\n{traceback.format_exc()}  status: {out / name / "status.json"}'); sys.exit(1)
    A = load_apply(case, out) if C.stage_ok(out, 's6_apply') else None
    if A:
        b, f = A['baseline'], A['final']; K = A['key_metrics']
        r, t, w = K['rerouted_nets'], K['target_nets'], K['worst_net']
        print(f'\n=== {case["name"]}: before -> after (policy {A["policy"]}, {A["accepted"]}/{A["tried"]} reroutes kept) ===')
        print(f'  rerouted nets   {r["cc_before_fF"]:10.3f} -> {r["cc_after_fF"]:10.3f} {CU}   {r["change_pct"]:+.2f}%  '
              f'({r["n"]} nets, {r["nets_freed"]} freed of >=99% coupling)')
        print(f'  target nets     {t["cc_before_fF"]:10.3f} -> {t["cc_after_fF"]:10.3f} {CU}   {t["change_pct"]:+.2f}%  ({t["n"]} nets: {t["selection"]})')
        print(f'  worst net       {w["before"][1]:10.3f} -> {w["after"][1]:10.3f} {CU}   ({w["before"][0]} -> {w["after"][0]})')
        print(f'  design total    {b["cc_fF"]:10.3f} -> {f["cc_fF"]:10.3f} {CU}   {A["cc_change_pct"]:+.2f}%  (includes nets not touched)')
        print(f'  wire length     {b["wl_um"]:10.1f} -> {f["wl_um"]:10.1f} um   {A["wl_change_pct"]:+.2f}%')
        print(f'  vias            {b["vias"]:10d} -> {f["vias"]:10d}      {A["via_change"]:+d}')
        print(f'  final layout    {A["final_def"]}')
    print(f'\nfull report: {out / "report.md"}')


if __name__ == '__main__':
    main()
