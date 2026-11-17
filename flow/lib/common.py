"""Shared helpers: repo paths, case config, OpenROAD runner, per-stage status files."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FLOW = REPO / 'flow'
OR_DIR = FLOW / 'openroad'
SIG = ('SIGNAL', 'ANALOG')


class StageError(RuntimeError):
    """Raised with a human-readable message that names the failing step and its log file."""


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_case(name, cases_file=None):
    cases = json.loads(Path(cases_file or REPO / 'config/cases.json').read_text())
    if name not in cases:
        raise StageError(f'unknown case "{name}". Known: {", ".join(sorted(cases))}')
    c = dict(cases[name]); c['name'] = name
    c['def'] = str((REPO / c['def']).resolve())
    c['lefs'] = [str((REPO / l).resolve()) for l in c['lefs']]
    c['rules'] = str((REPO / c['rules']).resolve())
    for p in [c['def'], c['rules'], *c['lefs']]:
        if not Path(p).exists():
            raise StageError(f'case "{name}": missing input file {p}')
    return c


def lef_routing_layers(lefs):
    """Routing layer names in LEF order (technology section)."""
    import re
    out = []
    for l in lefs:
        t = Path(l).read_text(errors='replace')
        # layer definitions are "LAYER name" lines without ';' (pin geometry uses "LAYER name ;")
        heads = [(m.group(1), m.end()) for m in re.finditer(r'^[ \t]*LAYER[ \t]+([^\s;]+)[ \t]*\r?$', t, re.M)]
        for name, pos in heads:
            end = re.compile(r'^[ \t]*END[ \t]+' + re.escape(name) + r'\b', re.M).search(t, pos)
            body = t[pos:end.start()] if end else ''
            if re.search(r'^\s*TYPE\s+ROUTING\b', body, re.M) and name not in out:
                out.append(name)
    return out


def check_lef_def(case):
    """Fail early, with a plain message, when the LEFs lack technology layers or masters used by the DEF."""
    import re
    if not lef_routing_layers(case['lefs']):
        raise StageError('no routing LAYER found in the LEF files: the technology LEF (or a merged tech+cell LEF) must be in --lef, first')
    macros = set()
    for l in case['lefs']:
        macros |= set(re.findall(r'^\s*MACRO\s+(\S+)', Path(l).read_text(errors='replace'), re.M))
    t = Path(case['def']).read_text(errors='replace')
    i, j = t.find('\nCOMPONENTS'), t.find('END COMPONENTS')
    used = set(re.findall(r'^\s*-\s+\S+\s+(\S+)', t[i:j], re.M)) if i >= 0 and j > i else set()
    missing = sorted(used - macros)
    if missing:
        raise StageError(f'{len(missing)} cell/macro type(s) used in the DEF are not defined in the LEF files, e.g. '
                         f'{", ".join(missing[:8])}. Add the LEF file(s) that define them after the technology LEF '
                         f'(A.def tech.lef cells.lef ...). To find them: grep -rl "MACRO {missing[0]}" <folder> '
                         f'--include="*.lef"')
    lef_rules = set()
    for l in case['lefs']:
        lef_rules |= set(re.findall(r'^\s*VIARULE\s+(\S+)', Path(l).read_text(errors='replace'), re.M | re.I))
    used_rules = set(re.findall(r'\+\s*VIARULE\s+(\S+)', t))
    if used_rules - lef_rules:
        raise StageError(f'the DEF uses via rule(s) {", ".join(sorted(used_rules - lef_rules))} that the LEF files do not '
                         'define (VIARULE ... GENERATE): the technology LEF is probably a different version from the one '
                         'the layout was made with. Use that technology LEF.')
    if not re.search(r'^TRACKS\s', t, re.M):
        print('  note: the DEF has no TRACKS; they are made from the LEF routing pitches for each reroute')
    if 'ROUTED' not in t[t.find('\nNETS'):]:
        raise StageError('the DEF has no routed nets (no ROUTED wiring in NETS); the flow needs a routed layout')
    return len(used)


def def_net_name(def_file, net):
    r"""Net name as written in the DEF NETS section (bus bits may be written escaped, e.g. data\[3\])."""
    import re
    t = Path(def_file).read_text(errors='replace')
    i = t.find('\nNETS')
    names = set(re.findall(r'^\s*-\s+(\S+)', t[i:t.find('END NETS', i)], re.M)) if i >= 0 else set()
    plain = net.replace('\\', '')
    for cand in (net, plain, re.sub(r'([\[\]])', r'\\\1', plain)):
        if cand in names:
            return cand
    raise StageError(f'--normalize: net "{net}" not found in the DEF NETS section')


def plain_file(p):
    """A .gz input is decompressed once into out/.cache/<sha>/ inside this folder (or $LEFDEF_CACHE); OpenROAD
    reads plain text. Nothing is written outside the repository folder by default."""
    import gzip
    p = Path(p)
    if p.suffix != '.gz':
        return p
    out = Path(os.environ.get('LEFDEF_CACHE') or REPO / 'out' / '.cache') / sha256(p)[:16] / p.name[:-3]
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + '.part')
        with gzip.open(p, 'rb') as f, open(tmp, 'wb') as g:
            shutil.copyfileobj(f, g)
        tmp.replace(out)
    return out


def lef_files(items):
    """LEF paths from files, folders (every *.lef / *.lef.gz inside) or comma-separated lists; technology LEF first."""
    paths, explicit = [], set()
    for it in items:
        for s in str(it).split(','):
            if not s:
                continue
            p = Path(s).expanduser().resolve()
            if p.is_dir():
                found = sorted(list(p.glob('*.lef')) + list(p.glob('*.lef.gz')))
                if not found:
                    raise StageError(f'no *.lef or *.lef.gz file in folder {p}')
                paths += found
            elif p.exists():
                paths.append(p); explicit.add(str(plain_file(p)))
            else:
                raise StageError(f'missing input file {p}')
    paths = list(dict.fromkeys(str(plain_file(p)) for p in paths))
    # the technology LEF (routing LAYER definitions) must be read first; keep the given order otherwise
    tech = [p for p in paths if lef_routing_layers([p])]
    if len(tech) > 1:
        # several LEFs define routing layers (e.g. tech.lef plus a merged tech+cell LEF): one is used as the technology
        # LEF (a file named *tech*, else the first named file); the other named ones contribute only their MACRO/SITE
        # definitions; technology LEFs that only came from folders are skipped
        named = [p for p in tech if p in explicit] or tech
        pick = ([p for p in named if 'tech' in Path(p).name.lower()] or named)[:1][0]
        keep_sites = set(re.findall(r'^\s*SITE\s+(\S+)\s*$', Path(pick).read_text(errors='replace'), re.M))
        notes = []
        for p in tech:
            if p == pick:
                continue
            i = paths.index(p)
            if p in explicit:
                cells = cells_only_lef(p, keep_sites)
                paths[i] = cells
                notes.append(f'{Path(p).name}: its cell definitions only' if cells else f'{Path(p).name}: no cells, skipped')
            else:
                paths[i] = None; notes.append(f'{Path(p).name} (folder): skipped')
        paths = [p for p in paths if p]
        print(f'  note: {len(tech)} LEFs define routing layers; technology LEF = {Path(pick).name}; '
              + '; '.join(notes[:4]) + (' ...' if len(notes) > 4 else ''))
        tech = [pick]
    return tech + [p for p in paths if p not in tech]


def cells_only_lef(path, skip_sites=()):
    """Copy of a LEF with only its MACRO (and new SITE) blocks, in out/.cache/; None if it has no MACRO."""
    lines = Path(path).read_text(errors='replace').splitlines()
    keep, i = [], 0
    while i < len(lines):
        m = re.match(r'^\s*(MACRO|SITE)\s+(\S+)\s*$', lines[i])
        if m:
            j = i + 1
            end = re.compile(r'^\s*END\s+' + re.escape(m.group(2)) + r'\s*$')
            while j < len(lines) and not end.match(lines[j]):
                j += 1
            if m.group(1) == 'MACRO' or m.group(2) not in skip_sites:
                keep += lines[i:j + 1]
            i = j + 1
            continue
        if re.match(r'^\s*(VERSION|BUSBITCHARS|DIVIDERCHAR)\b', lines[i]):
            keep.append(lines[i])
        i += 1
    if not any(re.match(r'^\s*MACRO\s', l) for l in keep):
        return None
    out = Path(os.environ.get('LEFDEF_CACHE') or REPO / 'out' / '.cache') / sha256(path)[:16] / (Path(path).stem + '.cells.lef')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(keep + ['END LIBRARY']) + '\n')
    return str(out)


SKY130_LAYERS = {'li1', 'met1', 'met2', 'met3', 'met4', 'met5'}


from .defcells import sanitize as sanitize_def  # noqa: E402


def own_case(name, def_file, lefs, layers, rules, pool_nets, normalize='auto', strict=False):
    """Case from command-line paths (relative paths are taken from the current directory)."""
    if not lefs:
        raise StageError('give the LEF files after the DEF: python flow/run.py A.def B.lef (files or folders)')
    lef_paths = lef_files(lefs)
    from . import preflight   # OpenROAD reads the LEFs once; vias / via rules it rejects are left out of copies
    lef_paths, pnotes = preflight.fix_lefs(lef_paths)
    for n in pnotes:
        print(f'  note: {n}')
    rl = lef_routing_layers(lef_paths)
    if not layers:   # default: every routing layer except the lowest (local interconnect / metal1 pin layer)
        sig = [l for l in rl[1:] if not l.lower().startswith('top')]   # IHP TopMetal1/2: thick power metals
        if len(sig) < 2:
            raise StageError(f'could not infer --layers from the LEF routing layers {rl}; pass --layers, e.g. met1-met5')
        layers = f'{sig[0]}-{sig[-1]}'
        print(f'--layers not given: using {layers} (routing layers in LEF: {", ".join(rl)})')
    if not rules:
        if rl and set(rl) <= SKY130_LAYERS:
            rules = str(REPO / 'data/pdk/sky130hd.rcx_rules')
        else:   # no OpenRCX rules for this technology: measure coupling with the geometric proxy (um of parallel run)
            rules = 'proxy'
            print('--rules not given and no bundled OpenRCX rules for these layers: coupling is measured with the '
                  'geometric proxy (um of same-layer parallel run within 2 pitches; give --rules FILE for fF)')
    c = dict(name=name, lefs=lef_paths, layers=layers, rules=rules if rules == 'proxy' else str(Path(rules).expanduser().resolve()),
             pool_nets=pool_nets, split='user', domain='user')
    if not Path(def_file).expanduser().exists():
        raise StageError(f'missing input file {Path(def_file).expanduser().resolve()}')
    c['def'] = str(plain_file(Path(def_file).expanduser().resolve()))
    # DEF parts the LEFs do not define (cells, via rules, layers): a DEF copy without them, so the flow keeps running
    # then OpenROAD reads the copy; each item it still reports (DEF line number) is left out and the copy rewritten
    from .defcells import statement_at
    src, drop = c['def'], set()
    for _ in range(60):
        c['def'], notes, fixed = sanitize_def(src, lef_paths, drop)
        errs = preflight.check_def(lef_paths, c['def'])
        if not errs:
            break
        where = [statement_at(c['def'], int(m.group(1))) for l in errs for m in [re.search(r'\bat line (\d+)', l)] if m]
        if not where:   # no line number: leave out what the error names, e.g. "undefined NONDEFAULTRULE (X)"
            where = [('name', n) for l in errs if 'DEF parser returns' not in l
                     for n in re.findall(r'\(([^()\s]+)\)', l)]
        new = [w for w in where if w and w not in drop]
        if not new:
            for n in notes:
                print(f'  note: {n}')
            sep = chr(10) + '    '
            raise StageError(f'OpenROAD cannot read the DEF even after the repairs above. All {len(errs)} error line(s):'
                             + sep + sep.join(errs[:60]))
        drop |= set(new)
    else:
        sep = chr(10) + '    '
        raise StageError(f'OpenROAD still cannot read the DEF after leaving out {len(drop)} item(s); last errors:'
                         + sep + sep.join(errs[:60]))
    if fixed:
        c['fixed_nets'] = fixed
    for n in notes:
        print(f'  note: {n}')
    if normalize == 'auto':
        c['normalize'] = 'auto'
    elif normalize and normalize != 'off':
        net, _, times = normalize.rpartition(':')
        if not net or not times.isdigit():
            net, times = normalize, '2'
        c['normalize'] = dict(net=def_net_name(c['def'], net), times=int(times))
    if not strict:
        c['tolerate_baseline_drc'] = True
    for p in [c['def'], *([] if c['rules'] == 'proxy' else [c['rules']]), *c['lefs']]:
        if not Path(p).exists():
            raise StageError(f'missing input file {p}')
    return c


def openroad_bin():
    exe = os.environ.get('OPENROAD', 'openroad')
    if shutil.which(exe) is None and not Path(exe).exists():
        raise StageError(f'OpenROAD not found ("{exe}"). Install it or set OPENROAD=/path/to/openroad.')
    return exe


def run_openroad(script, env, log, python=False, args=(), threads=4, timeout=3600):
    """Run one OpenROAD job; raise StageError (with the log path) on a non-zero exit or a missing DONE marker."""
    cmd = [openroad_bin(), '-no_init', '-no_splash', '-exit']
    if python:
        cmd += ['-python', str(script), *map(str, args)]
    else:
        cmd += ['-threads', str(threads), str(script)]
    full_env = dict(os.environ, **{k: str(v) for k, v in env.items()})
    t0 = time.monotonic()
    with open(log, 'w') as f:
        rc = subprocess.run(cmd, env=full_env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
    text = Path(log).read_text(errors='replace')
    # OpenROAD can exit 0 after a Python traceback, so a DONE marker is required as well
    if rc != 0 or 'FLOW_DONE' not in text:
        tail = '\n'.join(text.splitlines()[-15:])
        raise StageError(f'OpenROAD job failed (exit {rc}): {script.name}\n  log: {log}\n  last lines:\n{tail}')
    return time.monotonic() - t0


class Stage:
    """Context manager writing out/<case>/<stage>/status.json: ok/failed, seconds, error, outputs."""

    def __init__(self, out, name):
        self.dir = Path(out) / name; self.dir.mkdir(parents=True, exist_ok=True); self.name = name
        self.info = {}

    def __enter__(self):
        self.t0 = time.monotonic()
        (self.dir / 'status.json').write_text(json.dumps(dict(stage=self.name, status='running'), indent=1))
        return self

    def __exit__(self, typ, val, tb):
        st = dict(stage=self.name, status='ok' if typ is None else 'failed', seconds=round(time.monotonic() - self.t0, 2), **self.info)
        if typ is not None:
            st['error'] = str(val)
        (self.dir / 'status.json').write_text(json.dumps(st, indent=1))
        return False


def stage_ok(out, name):
    p = Path(out) / name / 'status.json'
    return p.exists() and json.loads(p.read_text()).get('status') == 'ok'


def utility(row, base):
    """WU67 utility on extracted totals: coupling reduction - 0.25*WL increase - 0.05*via increase (relative)."""
    return ((base['cc_fF'] - row['cc_fF']) / base['cc_fF'] - .25 * (row['wl_um'] - base['wl_um']) / base['wl_um']
            - .05 * (row['vias'] - base['vias']) / base['vias'])
