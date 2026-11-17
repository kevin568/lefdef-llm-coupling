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
