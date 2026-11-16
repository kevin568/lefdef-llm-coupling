"""Shared helpers: repo paths, case config, OpenROAD runner, per-stage status files."""
import hashlib
import json
import os
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
