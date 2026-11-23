#!/bin/bash
# LLM scoring without a GPU: PyTorch (CPU build), transformers and peft into the flow's Python, and the 4-bit base
# model the adapter was trained on (unsloth/Qwen2.5-3B-Instruct-bnb-4bit, pinned revision) into local-tools/models/.
# flow/lib/scorer.py unpacks the 4-bit weights on the CPU into the exact values the GPU computes with (no bitsandbytes)
# and merges the LoRA adapter from models/ into them.
#
#   bash tools/install_llm_cpu.sh [FOLDER]   # default FOLDER: local-tools/ (made by tools/install_openroad_conda.sh)
#   source local-tools/env.csh               # (bash: env.sh), then:
#   python3 flow/run.py A.def B.lef C.lef --policy llm
#
# Needs about 3.5 GB free in FOLDER (model 2.1 GB, Python packages about 1.2 GB); about 13 GB of RAM while scoring.
# Nothing is written outside FOLDER. Running again resumes interrupted downloads. Everything is downloaded with curl
# (the system's HTTPS certificates, like git): the pinned wheels in tools/lock/llm-cpu-py311.txt (for the flow's
# Python 3.11 from tools/install_openroad_conda.sh; pip then installs offline) and the model from huggingface.co (or
# $HF_ENDPOINT for a mirror). Another Python version: pip downloads (PyPI, download.pytorch.org). Offline machine: put the wheels in a
# folder and set WHEELS=folder; copy the model files (list below) into FOLDER/models/Qwen2.5-3B-Instruct-bnb-4bit/.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DEST="${1:-$REPO/local-tools}"
mkdir -p "$DEST"
PREFIX=$(cd "$DEST" && pwd)
PY="$PREFIX/env-python/bin/python3"
[ -x "$PY" ] || PY=$(command -v python3)
[ -z "${FLOW_PY:-}" ] || PY="$FLOW_PY"   # flow/run.py installs into the Python it runs with
MODEL="$PREFIX/models/Qwen2.5-3B-Instruct-bnb-4bit"
REV=2672b588848527e456d320c11926c794539f47d5
HF="${HF_ENDPOINT:-https://huggingface.co}"
say() { printf '\n==> %s\n' "$*"; }

export TMPDIR="$PREFIX/tmp" HOME="$PREFIX/home" XDG_CACHE_HOME="$PREFIX/home/.cache" PIP_NO_CACHE_DIR=1
export HF_HOME="$PREFIX/home/.cache/huggingface" PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$TMPDIR" "$HOME" "$MODEL"

say "disk space in $PREFIX"
free_kb=$(df -Pk "$PREFIX" | awk 'NR==2 {print $4}')
have_kb=$(du -sk "$MODEL" | awk '{print $1}')
need_kb=$(( 3500000 - have_kb ))
awk -v f="$free_kb" -v n="$need_kb" 'BEGIN { if (n < 0) n = 0; printf "  free %.1f GB, still needed about %.1f GB\n", f/1048576, n/1048576 }'
if [ "$free_kb" -lt "$need_kb" ] && [ -z "${FORCE:-}" ]; then
  echo "  not enough space. Free some, or give another folder: bash tools/install_llm_cpu.sh /path/with/space"
  echo "  (FORCE=1 to try anyway)"; exit 1
fi

say "Python packages (CPU build of PyTorch) -> $("$PY" -c 'import sys; print(sys.prefix)')"
if "$PY" -c "import torch, transformers, peft, safetensors" > /dev/null 2>&1; then
  echo "  already installed"
elif [ -n "${WHEELS:-}" ]; then
  "$PY" -m pip install --no-index --find-links "$WHEELS" torch transformers peft safetensors accelerate
elif [ "$("$PY" -c 'import sys, platform; print("%d.%d-%s" % (sys.version_info[:2] + (platform.machine(),)))')" = 3.11-x86_64 ]; then
  # the tested wheel set (tools/lock/llm-cpu-py311.txt): curl downloads it (it trusts the system's certificates, like
  # git), pip installs offline -- pip never connects, so a network that re-signs HTTPS cannot break it
  WH="$PREFIX/wheels-llm"; mkdir -p "$WH"
  grep '^https://' "$REPO/tools/lock/llm-cpu-py311.txt" | while read -r line; do
    url="${line%%#*}"; sum="${line##*sha256=}"; f="$WH/$(basename "$url" | sed 's/%2B/+/g')"
    if [ -s "$f" ] && echo "$sum  $f" | sha256sum -c --quiet - > /dev/null 2>&1; then
      continue
    fi
    echo "  download $(basename "$f")"
    curl -fL --retry 5 -C - -o "$f.part" "$url"
    mv "$f.part" "$f"
    echo "$sum  $f" | sha256sum -c --quiet - || { echo "  checksum mismatch, removed: $f (run again)"; rm -f "$f"; exit 1; }
  done
  "$PY" -m pip install --no-index --find-links "$WH" "torch==2.7.1+cpu" "transformers==4.57.3" "peft==0.18.1" safetensors accelerate
  rm -rf "$WH"
else
  # torch from the CPU index (+cpu exists only there), its dependencies from PyPI
  "$PY" -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "torch==2.7.1+cpu"
  "$PY" -m pip install "transformers==4.57.3" "peft==0.18.1" "safetensors>=0.4" "accelerate>=0.34"
fi
"$PY" -c "import torch, transformers, peft; print('  torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"

say "base model Qwen2.5-3B-Instruct, 4-bit (2.1 GB) -> $MODEL"
for f in added_tokens.json config.json generation_config.json merges.txt special_tokens_map.json tokenizer.json \
         tokenizer_config.json vocab.json model.safetensors; do
  if [ -s "$MODEL/$f" ]; then echo "  have $f"; continue; fi
  echo "  download $f"
  curl -fL --retry 5 -C - -o "$MODEL/$f.part" "$HF/unsloth/Qwen2.5-3B-Instruct-bnb-4bit/resolve/$REV/$f"
  mv "$MODEL/$f.part" "$MODEL/$f"
done
say "checksums"
(cd "$MODEL" && sha256sum -c "$REPO/tools/lock/qwen2.5-3b-instruct-bnb-4bit.sha256")
rm -rf "$TMPDIR"

say "self-test: load the model with the default adapter and score two prompts on the CPU"
LLM_DEVICE=cpu "$PY" "$REPO/flow/lib/scorer.py" --selftest
echo
echo "done. In each new terminal: source $PREFIX/env.csh (bash: env.sh), then"
echo "  python3 flow/run.py A.def B.lef C.lef --policy llm"
