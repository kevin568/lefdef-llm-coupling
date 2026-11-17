# lefdef-llm-coupling

Evaluate an LLM that proposes **net reroutes to reduce coupling capacitance** on routed LEF/DEF layouts.
Every number comes from a real flow: OpenROAD ECO rerouting followed by OpenRCX parasitic extraction.

```
LEF/DEF ──► s1 import ──► s2 candidates ──► s3 score ──► s4 route ──► s5 extract ──► s6 report
            baseline       trees + guides     LLM / tool    OpenROAD ECO   OpenRCX       report.md
            DEF/SPEF/geo   + prompt text      ranking       per candidate  coupling, U   report.json
```

What you get per layout: for each high-coupling net, the model's best rerouting tree is routed by OpenROAD
(only that net is ripped up), the result is extracted with OpenRCX, and the report lists the net's own coupling
capacitance before and after, the change of the whole design's coupling, wire length and via changes, and whether
the reroute is legal (target connected, no new DRC violation, every other net untouched).

## Coupling scan without OpenROAD (`flow/lite.py`)

```bash
python3 flow/lite.py A.def tech.lef cells.lef            # seconds; pure Python 3.6+, nothing else needed
python3 flow/lite.py A.def tech.lef cells.lef --spef A.spef   # real coupling (fF) from StarRC / Calibre / OpenRCX
```

Ranks every routed signal net by coupling (same-layer parallel run within 2 pitches, or fF from a SPEF), lists each
net's top neighbours, and for the top nets suggests lower-coupling trees with an estimated reduction. Results in
`out/lite/<DEF name>/` (report.md, nets.csv, suggestions.csv, guides/). The ranking is reliable (Spearman 0.90-0.96
against OpenRCX on 6 public layouts); the estimated reduction is an optimistic upper bound, because nothing is
rerouted or re-extracted: confirm the suggested nets with the full flow below.

## Built-in router (`--router builtin`, `flow/lib/router.py`)

Reroutes normally use OpenROAD's detailed router. When it rejects a technology (for example the 2024 conda OpenROAD
on commercial FinFET PDKs with MIM capacitor layers or device-layer pins), `--router auto` (the default) switches to
the built-in router on the full design; `--router builtin` forces it. It is a simplified track-grid A* router in the
spirit of MAGICAL's anaroute: LEF widths / spacings / pitches / preferred directions, every other net's wires and the
cells' pins / OBS as obstacles, via pads checked for spacing, off-grid pin access with exact spacing checks, and the
policy's candidate tree as a guide (cost penalty outside its corridor). Only the target net is rewritten; a spacing
self-check writes drc.rpt, so a reroute with a violation is not counted as valid. It is not a signoff DRC.

## Quick start

```bash
git clone https://github.com/kevin568/lefdef-llm-coupling.git    # the full URL (git clone owner/name alone fails)
cd lefdef-llm-coupling
pip install -r requirements.txt          # Python 3.10+; torch/peft/bitsandbytes only needed for the LLM
export OPENROAD=/path/to/openroad         # or have `openroad` on PATH
python flow/run.py tempsense                          # LLM policy (CUDA GPU), best tree per net
python flow/run.py tempsense --policy heuristic       # no GPU: geometric heuristic ranks the trees
python flow/run.py tempsense --route all              # route every candidate: ranking quality vs measured truth
cat out/tempsense/report.md
```

**Your own layout**: give the routed DEF, then its LEF files or folders. Nothing else is needed.

```bash
python flow/run.py A.def B.lef               # B.lef: technology + cells, or several files / folders
python flow/run.py A.def tech.lef macros/    # a folder means every *.lef in it
python flow/run.py A.def B.lef --nets vco    # only reroute nets whose name matches (regular expression)
```

What is decided for you (each can be overridden):
- **Inputs**: `.def.gz` and `.lef.gz` are accepted (decompressed once into `out/.cache/` in this folder); a folder
  means every `*.lef` / `*.lef.gz` in it.
- **LEF order**: the technology LEF (the one with routing `LAYER` definitions) is read first. Several technology LEFs
  (e.g. a whole PDK `lef/` folder) stop with a message: give one technology LEF plus its cell LEF.
- **Routing layers** (`--layers`): every LEF routing layer except the lowest, and without IHP's `TopMetal*` power
  layers (SKY130: `met1-met5`; IHP: `Metal2-Metal5`).
- **Extraction rules** (`--rules`): the bundled SKY130 file when the LEF layers are SKY130's (`li1`, `met1`-`met5`).
  Any other technology without `--rules FILE` uses the **geometric coupling proxy** (`--rules proxy`): coupling is
  the length (um) a net runs parallel to another net on the same layer within 2 routing pitches, written as a SPEF so
  every stage is unchanged; reroutes, DRC and the other-net checks are real OpenROAD runs, and reports say `um (proxy)`.
  Checked against OpenRCX afterwards (base.def vs final.def extracted with the real rules): Nangate45 gcd, proxy
  -61.6% on 14 rerouted nets, OpenRCX -49.3% (12 of 14 nets lower, design -12.7%); tempsense, proxy -30.3% on 9 nets,
  OpenRCX -41.1% (3 of 9 nets lower, design -8.9%). For fF on your technology, extract the input DEF and
  `s6_apply/final.def` with your extractor (StarRC, Calibre, Quantus) or pass `--rules FILE`.
- **Policy** (`--policy`): the LLM when a CUDA GPU and its packages are present, otherwise the geometric heuristic.
- **Published DEFs that OpenROAD re-patches** (`--normalize auto`): on some older layouts the first OpenROAD detailed
  route also fixes pre-existing defects, so every reroute would change other nets and be marked invalid. Stage
  `s1_import` reroutes, one at a time and with the router default (as the later stages do), the likely target nets
  plus one short 2-pin net elsewhere, and compares every other net. If nothing else changed (e.g. `tempsense`), the
  DEF is used as it is (and these reroutes are reused by `s2_candidates`). Otherwise the re-patched nets are rerouted
  into a new baseline and the check is repeated (at most 4 checks); the terminal prints a note, and says so when the
  DEF never becomes stable (then the DEF usually does not match the LEFs and few reroutes are valid).
  `s1_import/normalize.json` records every round. `--normalize off` disables it; `--normalize NET` reroutes a net
  you choose.
- **Pre-existing DRC violations**: a violation that every router-default reroute reports (or that the last, stable
  normalize round reports) is listed in `s2_candidates/baseline_violations.json`, printed as a count by type, and not
  counted as new. `--strict` counts it.
- **Output folder**: `out/<DEF name>`. A folder keeps an `inputs.json` fingerprint; a second run with a different
  layout or options in the same folder stops instead of reusing old stages (e.g. two `user_project_wrapper.def`):
  use `--out`.
- `s0_check` stops with a plain message if a cell/macro used by the DEF is missing from the LEFs, if the DEF uses a
  via rule (`VIARULE ... GENERATE`) the technology LEF does not define (a different PDK version: e.g. IHP's
  `viagen56` exists in the OpenROAD-flow-scripts copy of `sg13g2_tech.lef` but not in newer IHP-Open-PDK releases),
  or if the DEF is not routed.
- **DEF without `TRACKS`**: track grids are made from the LEF routing pitches for each reroute (a note is printed).
- **A net OpenROAD cannot reroute** (the router stops with an error on it) is left out of the pool with a note and
  its log path, instead of stopping the run.

Checked with nothing but `A.def B.lef` (plus `--nets`): `tempsense` keeps its published DEF (same baseline sha256
as the bundled case); the raw VCO-ADC v2 top level and `vco_macro` are detected as re-patched, patched
automatically, and give valid reroutes (`vco_macro`: the same tolerated met2 violation as its `cases.json` entry).

When the run ends, the terminal prints the key numbers, and `out/A/report.md` holds the full report:

```
=== vco_macro: before -> after (policy llm, 6/14 reroutes kept) ===
  rerouted nets       89.283 ->      0.669 fF   -99.25%  (6 nets, 4 freed of >=99% coupling)
  target nets        644.248 ->    520.638 fF   -19.19%  (20 nets: top 20 by coupling)
  worst net          260.185 ->    260.185 fF   (vco_enb\[1\] -> vco_enb\[1\])
  design total     58687.719 ->  58610.505 fF   -0.13%  (includes nets not touched)
  wire length      1046431.3 ->  1046675.2 um   +0.02%
  vias                  3437 ->       3439      +2
  final layout    out/vco_macro/s6_apply/final.def
```
(measured on the analog `vco_macro` top level). The improved layout is `s6_apply/final.def` with its parasitics in
`final.spef`.

Headline metrics, in order:
- **rerouted nets**: coupling of the nets whose reroute was kept, before → after. This is what the flow changed.
- **target nets**: the nets the flow examined (the `pool_nets` highest-coupling nets, or your `--nets` selection,
  e.g. the analog nets you care about).
- **worst net**: the largest coupling among the target nets.
- **design total**: every coupling capacitor in the layout, including nets the flow never touches (e.g. long
  chip-level IO lines with no better route), so it is a context row.

Measured on a 20-core machine with an RTX 5070 and `--jobs 4`: `tempsense` takes 2 minutes with `--route best`
(including the combining stage `s6_apply`) and 4.6 minutes with `--route all`; `vco_macro` takes 9 minutes with
`--route best` before `s6_apply`. Example (`tempsense`, LLM, `--route best`): 8 of 9 reroutes combined, total coupling
52.193 → 50.236 fF (−3.75%), wire length −0.7%, +3 vias.

**Check your installation** with `tempsense`: the baseline must read total coupling `52.193 fF`, wire length
`2115.7 um`, `756` vias; `s2_candidates` must report 20 nets and 212 candidates (192 trees + 20 router defaults).
With `--route all` all 192 trees are valid and 112 of them improve U.

### Requirements
- Linux (or WSL2). OpenROAD with Python (`odb`) support. Tested with OpenROAD `26Q2-1164-g08f67ee5ec` and with the
  conda build `2.0-12381` (Feb 2024). Stage `s0_check` verifies the commands the flow needs.
- **No OpenROAD, no root, older Linux** (e.g. RHEL/Rocky 8, glibc 2.28, no Docker): `bash tools/install_openroad_conda.sh`
  puts OpenROAD and the flow's Python into `local-tools/` inside this folder (about 1.9 GB, 3 GB while installing;
  downloads only, conda-forge + litex-hub, no Anaconda "defaults" channel). Nothing is written outside this folder:
  conda's cache, temporary files and home-directory files are redirected into `local-tools/`, which `.gitignore`
  keeps out of `git pull`. Then, in each new terminal, `source local-tools/env.csh` (csh/tcsh; bash: `env.sh`; check with `echo $0`) and run `python3 flow/run.py A.def B.lef --policy heuristic` (`python3`: `python` may be Python 2 or an alias). Running the script again skips what is installed.
  Package versions are pinned in `tools/lock/` (the tested set); curl downloads them (the same way git connects) and
  conda installs from the local files offline, so conda's own HTTPS checks (`CondaSSLError` behind company HTTPS
  inspection) are not involved. Every library
  it loads is inside that folder (needs glibc 2.17 or newer); checked end to end on `tempsense` (40/40 valid reroutes,
  same baseline as OpenROAD 26Q2). With that older OpenROAD, the geometry export reads wires from the DEF
  (`flow/lib/defwires.py`, identical shapes to OpenDB's on 6 checked layouts).
- LLM policy: a CUDA GPU with at least 8 GB. The base model `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (about 2 GB)
  is downloaded from Hugging Face on first use. Read `NOTICE.md` for its license.

## Layouts (`config/cases.json`)

All are public, OpenLane- or OpenROAD-routed layouts, copied unmodified (`data/SOURCES.json` pins URL, commit and hash).

| case | split | what it is |
|---|---|---|
| `vco_macro` | test | caravel VCO-ADC top level: analog VCO macro outputs routed to the ADC readout (candidate nets restricted to VCO macro pins) |
| `tempsense`, `digital_pll`, `vco_adc`, `audiodac` | test | standard-cell blocks of mixed-signal projects (temperature sensor, PLL, VCO-ADC phase readout, audio DAC) |
| `adc_clkgen`, `config_reg_mux` | test | digital blocks |
| `caravel_clocking`, `spare_logic_block`, `mprj_logic_high`, `const_gen`, `housekeeping`, `gcd_nangate45` | train | layouts used to fine-tune `models/llm_wu121_ft` |

To run your own layout, either use `python flow/run.py A.def B.lef` (Quick start) or add an entry to `config/cases.json`: `def`, `lefs` (technology LEF first), `layers`
(signal routing range, e.g. `met1-met5`), `rules` (OpenRCX rules for the technology), `pool_nets` (how many
high-coupling nets to try).

## Stages, outputs and debugging

Each stage writes `out/<case>/<stage>/status.json` (`ok` or `failed`, runtime, error text) and keeps every
OpenROAD log next to its output. When a stage fails, the command prints the stage name, the failing log path and
its last lines. Fix the cause and resume with `--from <stage>`; finished stages are skipped.

| stage | does | key outputs | if it fails, look at |
|---|---|---|---|
| `s0_check` | OpenROAD commands, GPU / Python packages, input hashes vs `data/SOURCES.json` | `status.json` | missing tool, wrong file (e.g. CRLF line endings after a Windows checkout) |
| `s1_import` | read LEF/DEF, rewrite baseline DEF, extract baseline SPEF, export geometry | `base.def`, `base.spef`, `geo.json`, `baseline.json` | `import.log`, `extract.log`, `export.log` |
| `s2_candidates` | pick the `pool_nets` nets with the largest extracted coupling, run OpenROAD's own reroute of each (pin access + "router default" candidate), build candidate trees, route guides and prompt text | `candidates.json`, `pin_access/*/route.log` | a `pin_access/*/route.log` |
| `s3_score` | LLM score (`logit Yes - logit No`) for every tree, plus heuristic and random scores | `scores.json` | GPU / model download |
| `s4_route` | OpenROAD ECO: rip up one net, read the candidate's guides, `detailed_route` | `runs/<id>/out.def`, `drc.rpt`, `route.log` | `runs/<id>/route.log`; failures are listed in `routed.json` |
| `s5_extract` | OpenRCX on every routed DEF; legality checks; coupling metrics | `runs/<id>/out.spef`, `metrics.json`, `measurements.json` | `runs/<id>/extract.log` |
| `s6_apply` | combine: per net the policy's top-scored routed tree with single-net U > 0, rerouted one after another on the evolving DEF (largest U first); a step is kept only if it is legal against the current layout and improves U again | `final.def`, `final.spef`, `apply.json`, `steps/<n>_<net>/` | `steps/*/route.log`; skipped steps are listed with reasons |
| `s7_report` | before/after of the combined layout, per-net single-reroute table, ranking metrics (`--route all`), invalid reroutes with reasons and run dirs | `report.md`, `report.json` | — |

Run a single stage with `--only <stage>`. Everything a stage reads comes from earlier stage folders, so any
intermediate file can be opened in OpenROAD or a text editor.

## Metrics
- **Coupling capacitance (CC)**: OpenRCX, `-coupling_threshold 0.1` fF (smaller couplings are grounded).
  Design CC counts each coupling capacitor once; a net's CC credits every coupling capacitor to both of its nets.
- **U** = (CC₀ − CC)/CC₀ − 0.25·(WL − WL₀)/WL₀ − 0.05·(vias − vias₀)/vias₀, against the baseline design.
- **Valid**: target net routed, no DRC violation beyond the case's `known_violations`, and the DEF text of every
  other net is unchanged.
- The LLM prompt describes geometry with a proxy (same-layer parallel run length within 2 pitches, µm); it is an
  input feature, not a reported metric.

## Models and reference results

See `models/README.md` and `docs/RESULTS.md`. In short, on the seven held-out test layouts:
- The rerouting flow is legal on 1,574 of the 1,576 candidate routes (99.9%): 1,516/1,516 on the six standard-cell
  layouts, 58/60 on `vco_macro`.
- On the analog `vco_macro` top level, the four VCO nets the fine-tuned LLM chose in a four-round loop drop from
  536.5 fF to 66.8 fF of coupling (−87.5%); single VCO phase/enable lines (e.g. `vco_enb[2]`, 248 fF) can be
  freed of coupling entirely.
- As a ranker, the default `llm_v2` adapter (Spearman 0.15 vs measured U) is ahead of random (0.02) but behind
  the simple geometric heuristic (0.32). `docs/RESULTS.md` gives every layout.

## Notes
- `vco_macro`: OpenROAD's detailed router patches pre-existing defects at two VCO macro pins the first time it runs
  on the published DEF, so the baseline is that DEF after two router-default reroutes of one net, and one
  pre-existing met2 spacing violation at a VCO pin is tolerated in every run (`config/cases.json`, `note`).
- Candidate generation is seeded by the net name, so candidates, guides and prompts are reproducible. The 4-bit
  LLM scores are not bit-exact between runs (on `tempsense`, two runs differed by up to 0.19), so a net whose top
  trees score almost the same can get a different pick; `--policy heuristic` is deterministic.
