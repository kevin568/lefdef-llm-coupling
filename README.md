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

## Quick start

```bash
pip install -r requirements.txt          # Python 3.10+; torch/peft/bitsandbytes only needed for the LLM
export OPENROAD=/path/to/openroad         # or have `openroad` on PATH
python flow/run.py tempsense                          # LLM policy (CUDA GPU), best tree per net
python flow/run.py tempsense --policy heuristic       # no GPU: geometric heuristic ranks the trees
python flow/run.py tempsense --route all              # route every candidate: ranking quality vs measured truth
cat out/tempsense/report.md
```

**Your own layout** (routed DEF + LEFs; no config file needed):

```bash
python flow/run.py A --def A.def --lef B.lef                        # B.lef = technology + every cell/macro used in A.def
python flow/run.py A --def A.def --lef tech.lef,cells.lef,macro.lef  # several LEFs: comma-separated, technology first
python flow/run.py A --def A.def --lef B.lef --nets "vco|clk"        # only reroute nets whose name matches
```

- `--layers` (signal routing range) defaults to every LEF routing layer except the lowest (SKY130: `met1-met5`).
- `--rules` (OpenRCX rules) defaults to the bundled SKY130 file for `met*` layers; pass it for other technologies.
- `s0_check` stops with a plain message if the LEFs lack technology layers or a cell/macro used by the DEF, or if
  the DEF is not routed.

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
- Linux (or WSL2). OpenROAD with Python (`odb`) support. Tested with OpenROAD `26Q2-1164-g08f67ee5ec`.
  Stage `s0_check` verifies the commands the flow needs.
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

To run your own layout, either use `--def/--lef/--layers` (Quick start) or add an entry to `config/cases.json`: `def`, `lefs` (technology LEF first), `layers`
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
- The rerouting flow is legal on 100% of the 1,634 candidate routes (`vco_macro`: 58/60).
- On the analog `vco_macro` top level, the four VCO nets the fine-tuned LLM chose in a four-round loop drop from
  536.5 fF to 66.8 fF of coupling (−87.5%); single VCO phase/enable lines (e.g. `vco_enb[2]`, 248 fF) can be
  freed of coupling entirely.
- As a ranker, the default `llm_v2` adapter (Spearman 0.15 vs measured U) is ahead of random (0.02) but behind
  the simple geometric heuristic (0.32). `docs/RESULTS.md` gives every layout.

## Notes
- `vco_macro`: OpenROAD's detailed router patches pre-existing defects at two VCO macro pins the first time it runs
  on the published DEF, so the baseline is that DEF after two router-default reroutes of one net, and one
  pre-existing met2 spacing violation at a VCO pin is tolerated in every run (`config/cases.json`, `note`).
- Candidate generation is seeded by the net name, so candidates, guides and prompts are reproducible. LLM scores
  reproduce to within about 0.01 across GPUs and batch composition.
