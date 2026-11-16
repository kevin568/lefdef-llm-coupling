# Reference results (seven held-out SKY130 layouts)

Produced with the same candidate generator, guides, router, extraction and metrics as `flow/` (the release flow
reproduces the research run exactly on `tempsense`: identical candidates, guides, prompts, legality and U;
LLM scores within 0.01). The test layouts were never used for training or model selection; model selection used
leave-one-layout-out on the train layouts only.

## Ranking quality (`--route all`): Spearman of the ranker's score vs measured U, valid candidates

| layout | candidates | `llm_v2` (default) | `llm_wu121_ft` seed 1 / 2 / 3 | heuristic | random |
|---|---:|---:|---:|---:|---:|
| vco_macro | 58 | 0.68 | 0.42 / −0.18 / 0.49 | 0.90 | 0.05 |
| tempsense | 192 | −0.04 | −0.01 / 0.08 / 0.28 | 0.35 | 0.05 |
| digital_pll | 294 | 0.01 | 0.04 / 0.06 / −0.10 | 0.16 | 0.02 |
| vco_adc | 275 | 0.27 | −0.24 / −0.23 / −0.02 | 0.12 | 0.10 |
| audiodac | 303 | 0.04 | −0.10 / −0.20 / −0.14 | 0.24 | 0.00 |
| adc_clkgen | 219 | 0.26 | 0.07 / 0.14 / 0.08 | −0.15 | −0.09 |
| config_reg_mux | 233 | −0.17 | −0.13 / 0.05 / −0.06 | 0.63 | 0.00 |
| **mean** | | **0.15** | 0.01 (mean of seeds) | **0.32** | 0.02 |

The shipped `models/llm_wu121_ft` is seed 1.

## Closed loop: 4 rounds × 3 routes, accept the best strict U improvement each round (design CC reduction)

| layout | `llm_wu121_ft` | `llm_v2` | heuristic | router default | random |
|---|---:|---:|---:|---:|---:|
| vco_macro | 0.50% | 0.48% | 0.48% | 0.46% | 0.52% |
| tempsense | 2.62% | 2.90% | 5.87% | 2.14% | 3.03% |
| digital_pll | 1.23% | 1.22% | 2.29% | 2.89% | 1.67% |
| vco_adc | 0.44% | 1.01% | 0.58% | 1.46% | 1.24% |
| audiodac | 0.40% | 0.44% | 0.26% | 0.15% | 0.34% |
| adc_clkgen | 0.44% | 0.66% | 0.21% | 0.40% | 0.35% |
| config_reg_mux | 5.40% | 0.84% | 8.73% | 3.40% | 3.62% |
| **mean** | 1.58% | 1.08% | 2.63% | 1.56% | 1.54% |

`vco_macro` design CC is dominated by long chip-level IO nets, so the design-level change is small. The coupling
of the rerouted VCO nets themselves: `llm_wu121_ft` 536.5 → 66.8 fF, `llm_v2` 322.8 → 0.3 fF, heuristic 322.2 → 0.4 fF,
router default 301.0 → 0.0 fF, random 552.6 → 68.0 fF. Rerouting the long VCO lines away from the bus they ran
beside removes most of their coupling whichever ranker picks them.

## Observations
- The frozen `llm_v2` score is sensitive to context in the prompt that does not change the candidate (circuit name,
  design-level totals): on `gcd_nangate45` the same candidates give Spearman 0.13 or −0.05 depending on it.
- Fine-tuning on about 1,100 LEF/DEF labels did not transfer to the held-out layouts; seed-to-seed variation
  exceeds the gain. The heuristic (`removed − newly added parallel length`) is the strongest single ranker here.
