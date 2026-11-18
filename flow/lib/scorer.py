"""LLM scorer: Qwen2.5-3B-Instruct (4-bit NF4, the base the adapter was trained on) + a LoRA adapter from models/.
Score = logit(Yes) - logit(No) at the last prompt token (same read-out as training).

GPU: bitsandbytes runs the 4-bit model (unsloth/Qwen2.5-3B-Instruct-bnb-4bit, downloaded from Hugging Face).
CPU: the same 4-bit checkpoint (tools/install_llm_cpu.sh puts it in local-tools/models/) is unpacked here into ordinary
float weights with the exact values bitsandbytes computes with, the adapter is merged in, and prompts of one net share
their common beginning (computed once). No bitsandbytes needed.
Environment: LLM_BASE_MODEL (folder or Hugging Face name), LLM_DEVICE (auto|cpu|cuda),
LLM_DTYPE (CPU: auto|float32|bfloat16|int8), LLM_THREADS (CPU threads; default: PyTorch's choice).

    python3 flow/lib/scorer.py --selftest                      # load, score two prompts, print speed
    python3 flow/lib/scorer.py --candidates out/X/s2_candidates/candidates.json --out scores.json
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
REPO = Path(__file__).resolve().parents[2]
GPU_BASE = 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit'
LOCAL_BASE = REPO / 'local-tools' / 'models' / 'Qwen2.5-3B-Instruct-bnb-4bit'
DEFAULT_ADAPTER = REPO / 'models' / 'llm_v2' / 'adapter'
SYSTEM = 'You are an analog layout routing assistant. Answer with one word.'


def device():
    want = os.environ.get('LLM_DEVICE', 'auto')
    if want != 'auto':
        return want
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def base_model(dev):
    if os.environ.get('LLM_BASE_MODEL'):
        return os.environ['LLM_BASE_MODEL']
    if dev == 'cpu' or (LOCAL_BASE / 'config.json').exists():
        return str(LOCAL_BASE)
    return GPU_BASE


def available():
    """(ok, device, why): can --policy llm run here?"""
    try:
        import torch, transformers, peft, safetensors  # noqa: F401
    except Exception as e:
        return False, None, f'LLM packages missing ({e}); CPU: bash tools/install_llm_cpu.sh'
    dev = device()
    if dev == 'cuda':
        if not torch.cuda.is_available():
            return False, dev, 'LLM_DEVICE=cuda but no CUDA GPU'
        try:
            import bitsandbytes  # noqa: F401
        except Exception as e:
            return False, dev, f'the 4-bit model on a GPU needs bitsandbytes ({e})'
        return True, dev, torch.cuda.get_device_name()
    b = Path(base_model(dev))
    if not (b / 'config.json').exists():
        return False, dev, f'no GPU and no local model in {b}; run: bash tools/install_llm_cpu.sh'
    return True, dev, f'CPU, {b}'


def mem_available_gb():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return 0.


def dequant_nf4(f, k, torch):
    """bitsandbytes NF4 with double quantization -> the bfloat16 weight bitsandbytes multiplies with."""
    st = json.loads(bytes(f.get_tensor(k + '.quant_state.bitsandbytes__nf4').tolist()).decode())
    packed = f.get_tensor(k).flatten()
    nmap, nabs = f.get_tensor(k + '.nested_quant_map'), f.get_tensor(k + '.nested_absmax')
    aq = f.get_tensor(k + '.absmax').long(); nb = st['nested_blocksize']
    pad = (-len(aq)) % nb
    am = nmap[torch.nn.functional.pad(aq, (0, pad))].view(-1, nb) * nabs.view(-1, 1)
    absmax = am.flatten()[:len(aq)] + st['nested_offset']
    codes = torch.stack([packed >> 4, packed & 15], 1).flatten().long()   # high nibble first
    qmap = f.get_tensor(k + '.quant_map'); bs = st['blocksize']; n = 1
    for d in st['shape']:
        n *= d
    vals = qmap[codes]
    vals = torch.nn.functional.pad(vals, (0, (-len(vals)) % bs)).view(-1, bs) * absmax[:(len(vals) + bs - 1) // bs].view(-1, 1)
    return vals.flatten()[:n].view(*st['shape']).to(getattr(torch, st['dtype']))


def load_dequantized(base, dtype, torch):
    """The 4-bit checkpoint as an ordinary float model on the CPU (weights = bitsandbytes' dequantized values)."""
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(base)
    if hasattr(cfg, 'quantization_config'):
        del cfg.quantization_config
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
    sd = {}
    for fn in sorted(Path(base).glob('*.safetensors')):
        with safe_open(str(fn), 'pt') as f:
            keys = list(f.keys()); quant = {k[:-len('.quant_state.bitsandbytes__nf4')] for k in keys if k.endswith('.bitsandbytes__nf4')}
            for k in keys:
                if k in quant:
                    sd[k] = dequant_nf4(f, k, torch).to(dtype)
                elif not any(k.startswith(q + '.') for q in quant):
                    sd[k] = f.get_tensor(k).to(dtype)
    res = model.load_state_dict(sd, strict=False, assign=True)
    if res.unexpected_keys or [k for k in res.missing_keys if k != 'lm_head.weight']:
        raise RuntimeError(f'4-bit checkpoint does not match the model: missing {res.missing_keys[:5]}, unexpected {res.unexpected_keys[:5]}')
    model.tie_weights()
    rot = model.model.rotary_emb   # buffers not in the checkpoint: build them for real
    model.model.rotary_emb = type(rot)(config=cfg)
    meta = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    if meta:
        raise RuntimeError(f'not loaded: {meta[:5]}')
    return model


def is_4bit(base):
    c = Path(base) / 'config.json'
    return c.exists() and 'bitsandbytes' in json.dumps(json.loads(c.read_text()).get('quantization_config', {}))


class Scorer:
    def __init__(self, adapter_dir=None, log=print):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        self.torch, self.log = torch, log
        self.dev = device(); base = base_model(self.dev)
        adapter_dir = Path(adapter_dir or DEFAULT_ADAPTER)
        t0 = time.time()
        if self.dev == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('LLM_DEVICE=cuda but no CUDA GPU')
            from peft import prepare_model_for_kbit_training
            model = AutoModelForCausalLM.from_pretrained(base, device_map={'': 0})
            model.eval(); model.config.use_cache = False
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
            self.dtype = 'bnb-4bit'
        else:
            if os.environ.get('LLM_THREADS'):
                torch.set_num_threads(int(os.environ['LLM_THREADS']))
            want = os.environ.get('LLM_DTYPE', 'auto')
            if want == 'auto':   # float32 needs about 13 GB of RAM; bfloat16 half of that (slower on CPUs without bf16 units)
                want = 'float32' if mem_available_gb() >= 16 or mem_available_gb() == 0 else 'bfloat16'
            self.dtype = want
            dt = torch.float32 if want == 'int8' else getattr(torch, want)
            model = load_dequantized(base, dt, torch) if is_4bit(base) else \
                AutoModelForCausalLM.from_pretrained(base, torch_dtype=dt, low_cpu_mem_usage=True)
            model.eval(); model.config.use_cache = False
        self.tok = AutoTokenizer.from_pretrained(base)
        yes = self.tok.encode('Yes', add_special_tokens=False); no = self.tok.encode('No', add_special_tokens=False)
        assert len(yes) == len(no) == 1
        self.yes, self.no = yes[0], no[0]
        wrapper = get_peft_model(model, PeftConfig.from_pretrained(str(adapter_dir)))
        state = load_file(str(adapter_dir / 'adapter_model.safetensors'), device='cuda:0' if self.dev == 'cuda' else 'cpu')
        res = set_peft_model_state_dict(wrapper, state, adapter_name='default')
        if res.unexpected_keys or any('lora_' in k for k in res.missing_keys):
            raise RuntimeError(f'adapter does not match the base model: {adapter_dir}')
        wrapper.eval()
        # CPU: fold the LoRA update into the weights (same function, no extra matmuls per layer)
        self.model = wrapper.get_base_model() if self.dev == 'cuda' else wrapper.merge_and_unload()
        w = self.model.get_output_embeddings().weight
        self.dvec = (w[self.yes].float() - w[self.no].float()).detach().clone()
        if self.dev == 'cpu' and self.dtype == 'int8':   # dynamic int8 matmuls in the transformer body (read-out stays float)
            torch.ao.quantization.quantize_dynamic(self.model.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        self.identity = dict(base_model=base, adapter=str(adapter_dir), adapter_tensors=len(state), device=self.dev,
                             dtype=self.dtype, threads=torch.get_num_threads() if self.dev == 'cpu' else None,
                             load_s=round(time.time() - t0, 1))

    def _ids(self, text):
        t = self.tok.apply_chat_template([dict(role='system', content=SYSTEM), dict(role='user', content=text)],
                                         tokenize=False, add_generation_prompt=True)
        return self.tok(t, return_tensors='pt').input_ids[0]

    def _progress(self, done, total):
        now = time.time()
        if self.log and (now - self._last > 20 or done == total):
            rate = (now - self._t0) / done
            self.log(f'  LLM scored {done}/{total} prompts, {rate:.2f} s/prompt, about {rate * (total - done) / 60:.1f} min left', flush=True)
            self._last = now

    def score(self, texts, bs=8, groups=None):
        """groups: one key per text (e.g. the net); on the CPU, texts of one group share their common beginning."""
        ids_all = [self._ids(t) for t in texts]
        self._t0 = self._last = time.time()
        with self.torch.inference_mode():
            out = self._score_gpu(ids_all, bs) if self.dev == 'cuda' else self._score_cpu(ids_all, bs, groups)
        self.seconds = time.time() - self._t0
        return out

    def _score_gpu(self, ids_all, bs):   # as in training/evaluation: original order, left padding, bf16 autocast
        torch = self.torch; out = []; pad = self.tok.pad_token_id or 0
        for i in range(0, len(ids_all), bs):
            ids = ids_all[i:i + bs]; n = max(len(x) for x in ids)
            inp = torch.full((len(ids), n), pad, dtype=torch.long); mask = torch.zeros((len(ids), n), dtype=torch.long)
            for k, x in enumerate(ids):
                inp[k, n - len(x):] = x; mask[k, n - len(x):] = 1
            pos = (mask.cumsum(1) - 1).clamp(min=0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = self.model.model(input_ids=inp.to(0), attention_mask=mask.to(0), position_ids=pos.to(0),
                                     use_cache=False).last_hidden_state[:, -1, :]
            out.extend(float(v) for v in (h.float() @ self.dvec.to(h.device)))
            self._progress(len(out), len(ids_all))
        return out

    def _score_cpu(self, ids_all, bs, groups):
        from transformers import DynamicCache
        torch = self.torch; out = [None] * len(ids_all); pad = self.tok.pad_token_id or 0; done = 0
        by = {}
        for i, g in enumerate(groups or range(len(ids_all))):
            by.setdefault(g, []).append(i)
        self.shared_tokens = self.total_tokens = 0
        for idx in by.values():
            seqs = [ids_all[i] for i in idx]
            L = min(len(s) for s in seqs) - 1   # keep at least one token per prompt after the shared part
            for s in seqs[1:]:
                neq = (s[:L] != seqs[0][:L]).nonzero()
                L = int(neq[0]) if len(neq) else L
            legacy = None
            if L > 0:
                pre = self.model.model(input_ids=seqs[0][:L][None], use_cache=True)
                legacy = [(l.keys, l.values) for l in pre.past_key_values.layers] if hasattr(pre.past_key_values, 'layers') \
                    else pre.past_key_values.to_legacy_cache()
            self.shared_tokens += L * (len(seqs) - 1); self.total_tokens += sum(len(s) for s in seqs)
            order = sorted(range(len(idx)), key=lambda j: len(seqs[j]))
            for a in range(0, len(order), bs):
                js = order[a:a + bs]; suf = [seqs[j][L:] for j in js]; m = max(len(x) for x in suf); b = len(js)
                inp = torch.full((b, m), pad, dtype=torch.long); mask = torch.zeros((b, L + m), dtype=torch.long)
                for r, x in enumerate(suf):   # right padding: the prompts end at different positions
                    inp[r, :len(x)] = x; mask[r, :L + len(x)] = 1
                pos = (L + torch.arange(m))[None].expand(b, m)
                cache = DynamicCache.from_legacy_cache(tuple((k.expand(b, -1, -1, -1).contiguous(), v.expand(b, -1, -1, -1).contiguous())
                                                             for k, v in legacy)) if legacy else None
                h = self.model.model(input_ids=inp, attention_mask=mask, position_ids=pos, past_key_values=cache,
                                     use_cache=cache is not None).last_hidden_state
                last = torch.tensor([len(x) - 1 for x in suf])
                for r, j in enumerate(js):
                    out[idx[j]] = float(h[r, last[r]].float() @ self.dvec)
                done += b; self._progress(done, len(ids_all))
        return out


SAMPLE = ('Task: decide whether rerouting one net with the proposed tree improves layout utility U = coupling reduction - '
          '0.25*WL increase - 0.05*via increase (all relative to the original layout). Coupling = same-layer parallel run '
          'length to other nets within 2 pitches (um).\nCircuit selftest, block selftest: coupling 500.00 um, wirelength '
          '2000.0 um, vias 700.\nTarget net n1: 4 pins.\nCurrent route: 40.0 um of wire on M2H 21, M3V 19; coupling to other '
          'nets {cur} um (n2 {cur}).\nProposed tree: 5 segments, 41.0 um centreline; keeps 40% of the current path, adds '
          '24.0 um of new path.\nCoupling on abandoned path: {gone} um (n2 {gone}).\nNew path runs parallel to other nets on '
          'layers this net uses: {new} um (n3 {new}); on any layer: {new} um.\nWill this reroute improve U? Answer Yes or No.')


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--candidates', help='s2_candidates/candidates.json to score')
    ap.add_argument('--out', help='write {candidate_id: score} here')
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--no-share', action='store_true', help='CPU: do not share the common beginning of one net\'s prompts')
    a = ap.parse_args()
    ok, dev, why = available()
    print(f'device: {dev}  ({why})')
    if not ok:
        sys.exit(1)
    sc = Scorer(a.adapter)
    print('model:', json.dumps(sc.identity))
    if a.selftest:
        texts = [SAMPLE.format(cur='30.00', gone='28.00', new='1.00'), SAMPLE.format(cur='30.00', gone='2.00', new='25.00')]
        s = sc.score(texts, bs=2, groups=[0, 0])
        print(f'scores: removes most coupling {s[0]:+.3f}, adds coupling {s[1]:+.3f}  ({sc.seconds / 2:.2f} s/prompt)')
        print('selftest ok')
    if a.candidates:
        P = json.load(open(a.candidates))
        trees = [c for c in P['candidates'] if c['kind'] != 'default']
        s = sc.score([c['prompt'] for c in trees], bs=a.bs, groups=None if a.no_share else [c['net'] for c in trees])
        res = {c['candidate_id']: v for c, v in zip(trees, s)}
        json.dump(dict(model=sc.identity, seconds=sc.seconds, scores=res), open(a.out or 'scores.json', 'w'), indent=1)
        extra = f', {sc.shared_tokens}/{sc.total_tokens} tokens shared' if dev == 'cpu' else ''
        print(f'{len(res)} prompts in {sc.seconds:.0f} s -> {a.out or "scores.json"}{extra}')


if __name__ == '__main__':
    main()
