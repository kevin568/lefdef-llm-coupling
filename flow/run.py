#!/usr/bin/env python3
"""LEF/DEF -> candidate reroutes -> LLM ranking -> OpenROAD ECO route -> OpenRCX extraction -> coupling report.

  python flow/run.py tempsense                      # all stages, LLM policy, route the model's best tree per net
  python flow/run.py tempsense --policy heuristic   # no GPU needed
  python flow/run.py tempsense --route all          # route every candidate (ground truth for ranking metrics)
  python flow/run.py tempsense --from s4_route      # re-run from one stage (earlier stages must be ok)

Every stage writes out/<case>/<stage>/status.json and keeps every OpenROAD log next to its outputs.
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
from lib import spef, guides as G, candidates as K, features as F, proxy  # noqa: E402

STAGES = ['s0_check', 's1_import', 's2_candidates', 's3_score', 's4_route', 's5_extract', 's6_report']


def safe(name):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('_')[:40]


def def_nets(path):
    t = Path(path).read_text(errors='replace'); s = t[t.index('\nNETS'):t.index('END NETS')]
    out = {}
    for blk in s.split(';')[:-1]:
        blk = blk.strip(); m = re.match(r'-\s+(\S+)', blk)
        if m: out[m.group(1)] = ' '.join(blk.split())
    return out


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


def extract_job(case, def_in, spef_out, log, def_out=''):
    return C.run_openroad(C.OR_DIR / 'extract.tcl', dict(LEFS=' '.join(case['lefs']), DEF_IN=def_in, RULES=case['rules'],
                          SPEF_OUT=spef_out, DEF_OUT=def_out), log, threads=ARGS.threads)


def export_job(case, def_in, out_json, log):
    return C.run_openroad(C.OR_DIR / 'export_geo.py', {}, log, python=True, args=[','.join(case['lefs']), def_in, out_json])


def design_metrics(geo, sp):
    sig = [v for v in geo['nets'].values() if v['signal_type'] in C.SIG]
    return dict(cc_fF=sp['total_cc'], wl_um=sum(v['wl_dbu'] for v in sig) / geo['dbu'], vias=sum(v['vias'] for v in sig))


def pm_nets(geo):
    return {n: [(2 * geo['layers'][s['layer']]['level'], 0, s['rect']) for s in v['shapes']]
            for n, v in geo['nets'].items() if v['signal_type'] in C.SIG}


def net_ok(case, geo, n):
    v = geo['nets'][n]
    if v['signal_type'] not in C.SIG or not v['routed'] or not (2 <= len(v['pins']) <= 8):
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
    for p in [case['def'], case['rules'], *case['lefs']]:
        want = src.get(Path(p).resolve())
        if want and C.sha256(p) != want:
            raise C.StageError(f'input file differs from data/SOURCES.json: {p}\n  (line-ending conversion on checkout? see README "Debugging")')
    st.info['inputs_verified'] = True
    if ARGS.policy == 'llm':
        try:
            import torch, transformers, peft, bitsandbytes  # noqa: F401
        except Exception as e:
            raise C.StageError(f'LLM policy needs torch/transformers/peft/bitsandbytes ({e}); pip install -r requirements.txt or use --policy heuristic')
        if not torch.cuda.is_available():
            raise C.StageError('LLM policy needs a CUDA GPU; use --policy heuristic to run without it')
        st.info['gpu'] = torch.cuda.get_device_name()
    st.info['case'] = {k: case[k] for k in ('def', 'layers', 'rules', 'split', 'domain')}


def s1_import(case, out, st):
    d = st.dir
    extract_job(case, case['def'], d / 'input.spef', d / 'import.log', def_out=d / 'input_rewritten.def')
    base_def = d / 'input_rewritten.def'
    norm = case.get('normalize')
    if norm:   # deterministic DRT patching of pre-existing defects in the public DEF (see README, "vco_macro")
        cur = base_def
        for i in range(norm['times']):
            rd = d / f'normalize_{i + 1}'
            route_job(case, cur, [norm['net']], 'default', rd); cur = rd / 'out.def'
        base_def = cur
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
    if not eligible:
        raise C.StageError('no eligible net (signal, routed, 2..8 pins, case filter)')
    nets = sorted(eligible, key=lambda n: (-sp['listed_cc'].get(n, 0.), n))[:case.get('pool_nets', 20)]
    # pin access: OpenROAD's own reroute of each pool net (its global-route guide gives the pin-access gcells;
    # the same run is also the "default" (router's choice) candidate of that net)
    pdirs = {n: d / 'pin_access' / f'{j:02d}_{safe(n)}' for j, n in enumerate(nets)}
    with ThreadPoolExecutor(ARGS.jobs) as ex:
        list(ex.map(lambda n: route_job(case, base_def, [n], 'default', pdirs[n]), nets))
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
    known = [tuple(k) for k in case.get('known_violations', [])]

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


def s6_report(case, out, st):
    from statistics import mean
    M = json.loads((out / 's5_extract/measurements.json').read_text()); base = M['baseline']; rows = M['rows']
    S = {r['candidate_id']: r for r in json.loads((out / 's3_score/scores.json').read_text())['rows']}
    pol = 'llm' if ARGS.policy == 'llm' else ARGS.policy
    trees = [r for r in rows if r['kind'] != 'default']; dflt = {r['net']: r for r in rows if r['kind'] == 'default'}
    lines = [f'# Coupling report: {case["name"]}', '',
             f'Baseline: total coupling {base["cc_fF"]:.3f} fF, wire length {base["wl_um"]:.1f} um, vias {base["vias"]}, '
             f'{base["instances"]} instances, {base["signal_nets"]} signal nets.', '',
             f'Policy `{ARGS.policy}`, route mode `{ARGS.route}`. One net rerouted per row (OpenROAD ECO), measured by OpenRCX.', '',
             '| net | candidate | valid | net CC before → after (fF) | design ΔCC (fF) | U | router default: net CC after / U |',
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
                  f'rerouted nets coupling {summ["net_cc_before_fF"]:.2f} → {summ["net_cc_after_fF"]:.2f} fF.']
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


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case', help='case name from config/cases.json')
    ap.add_argument('--policy', default='llm', choices=['llm', 'heuristic', 'random'])
    ap.add_argument('--adapter', help='LoRA adapter dir (default models/llm_v2/adapter)')
    ap.add_argument('--route', default='best', choices=['best', 'all'], help='best: policy top-1 per net; all: every candidate')
    ap.add_argument('--from', dest='start', choices=STAGES, help='re-run from this stage')
    ap.add_argument('--only', choices=STAGES, help='run a single stage')
    ap.add_argument('--out', default=None, help='output dir (default out/<case>)')
    ap.add_argument('--jobs', type=int, default=3); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--cases', default=None, help='alternative cases.json')
    ARGS = ap.parse_args()
    try:
        case = C.load_case(ARGS.case, ARGS.cases)
    except C.StageError as e:
        print(f'ERROR: {e}'); sys.exit(2)
    out = Path(ARGS.out or C.REPO / 'out' / ARGS.case).resolve(); out.mkdir(parents=True, exist_ok=True)
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
    print(f'\nreport: {out / "report.md"}')


if __name__ == '__main__':
    main()
