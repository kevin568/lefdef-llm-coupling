"""LLM scorer: Qwen2.5-3B-Instruct (4-bit, downloaded from Hugging Face) + a LoRA adapter from models/.
Score = logit(Yes) - logit(No) at the last prompt token, batched with left padding (same read-out as training)."""
import os
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
BASE_MODEL = os.environ.get('LLM_BASE_MODEL', 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit')
SYSTEM = 'You are an analog layout routing assistant. Answer with one word.'


class Scorer:
    def __init__(self, adapter_dir):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
        from safetensors.torch import load_file
        if not torch.cuda.is_available():
            raise RuntimeError('the 4-bit LLM needs a CUDA GPU; use --policy heuristic to run the flow without it')
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map={'': 0})
        model.eval(); model.config.use_cache = False
        yes = self.tok.encode('Yes', add_special_tokens=False); no = self.tok.encode('No', add_special_tokens=False)
        assert len(yes) == len(no) == 1
        self.yes, self.no = yes[0], no[0]
        adapter_dir = Path(adapter_dir)
        base = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        wrapper = get_peft_model(base, PeftConfig.from_pretrained(str(adapter_dir)))
        state = load_file(str(adapter_dir / 'adapter_model.safetensors'), device='cuda:0')
        res = set_peft_model_state_dict(wrapper, state, adapter_name='default')
        if res.unexpected_keys or any('lora_' in k for k in res.missing_keys):
            raise RuntimeError(f'adapter does not match the base model: {adapter_dir}')
        wrapper.eval(); self.model = wrapper.get_base_model()
        self.identity = dict(base_model=BASE_MODEL, adapter=str(adapter_dir), adapter_tensors=len(state))

    def _ids(self, text):
        t = self.tok.apply_chat_template([dict(role='system', content=SYSTEM), dict(role='user', content=text)],
                                         tokenize=False, add_generation_prompt=True)
        return self.tok(t, return_tensors='pt').input_ids[0]

    def score(self, texts, bs=8):
        torch = self.torch; out = []
        pad = self.tok.pad_token_id or 0
        w = self.model.get_output_embeddings().weight
        dvec = (w[self.yes].float() - w[self.no].float())
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                ids = [self._ids(t) for t in texts[i:i + bs]]; n = max(len(x) for x in ids)
                inp = torch.full((len(ids), n), pad, dtype=torch.long); mask = torch.zeros((len(ids), n), dtype=torch.long)
                for k, x in enumerate(ids):
                    inp[k, n - len(x):] = x; mask[k, n - len(x):] = 1
                pos = (mask.cumsum(1) - 1).clamp(min=0)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    h = self.model.model(input_ids=inp.to(0), attention_mask=mask.to(0), position_ids=pos.to(0),
                                         use_cache=False).last_hidden_state[:, -1, :]
                out.extend(float(v) for v in (h.float() @ dvec))
        return out
