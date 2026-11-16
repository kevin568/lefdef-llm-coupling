# Models

Both are LoRA adapters (rank 16, q/k/v/o projections, all layers) for `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`.
The flow asks the model whether rerouting one net with a proposed tree improves
`U = coupling reduction - 0.25*WL increase - 0.05*via increase` and uses `logit(Yes) - logit(No)` as the score.

| folder | default | training data | notes |
|---|---|---|---|
| `llm_v2/adapter` | yes | 5,081 measured reroutes on six MAGICAL analog circuits (OTA, comparator, ADC), pairwise ranking loss with measured-feedback prompts, 400 steps | best LLM ranker on the seven held-out SKY130 layouts (`docs/RESULTS.md`) |
| `llm_wu121_ft/adapter` | no | same analog data + 1,136 measured LEF/DEF reroutes on six public layouts not in the test set, fresh LoRA, 200 steps | DEF-domain fine-tune; did not beat `llm_v2` on the held-out layouts |

Select one with `--adapter models/llm_wu121_ft/adapter`.
