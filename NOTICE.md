# Third-party material

The code in `flow/` is BSD-3-Clause (see `LICENSE`). Everything else keeps its own license.

## Layout data (`data/`)

Every file is listed in `data/SOURCES.json` with its upstream URL (pinned commit), license, git blob id and SHA-256.
The files are unmodified copies.

| upstream repository | license | files |
|---|---|---|
| efabless/caravel | Apache-2.0 | digital_pll, caravel_clocking, spare_logic_block, mprj_logic_high, housekeeping DEFs |
| iic-jku/mpw8-submission | Apache-2.0 | tempsense, audiodac, config_reg_mux, const_gen DEFs |
| duyhieubui/caravel_vco_adc | Apache-2.0 | vco_adc, vco_macro DEFs; VCO / VCO-ADC macro LEFs |
| efabless/SKY130_SAR-ADC1 | Apache-2.0 | adc_clkgen DEF; merged SKY130 technology + standard-cell LEF |
| efabless/sky130_sram_macros | Apache-2.0 | sram.lef |
| The-OpenROAD-Project/OpenROAD | BSD-3-Clause | gcd (Nangate45) DEF, Nangate45 LEF, OpenRCX rule files |

## Models (`models/`)

`models/*/adapter` are LoRA adapters (PEFT) trained by us. They are applied on top of
`unsloth/Qwen2.5-3B-Instruct-bnb-4bit`, a 4-bit quantisation of `Qwen/Qwen2.5-3B-Instruct`, which is **not**
redistributed here: it is downloaded from Hugging Face at run time. Use of the base model is governed by the
Qwen license (https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE, a research licence that restricts
commercial use). The adapters are provided for research use under the same conditions.

## Tools

OpenROAD (BSD-3-Clause) is required at run time and is not bundled.
