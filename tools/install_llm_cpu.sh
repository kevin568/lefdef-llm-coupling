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
# (the system's HTTPS certificates, like git), and it first prints which download sites this machine reaches. The
# flow's conda Python 3.11 (tools/install_openroad_conda.sh): the pinned conda-forge packages in
# tools/lock/llm-cpu-conda-py311.txt (conda.anaconda.org; conda installs offline). Another Python 3.11: the pinned
# wheels in tools/lock/llm-cpu-py311.txt (download.pytorch.org, files.pythonhosted.org; pip installs offline). Other
# versions: pip downloads. The model comes from huggingface.co (or $HF_ENDPOINT for a mirror). Offline machine: put the wheels in a
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

say "which download sites this machine reaches (curl, 10 s each; a company firewall may block some)"
for h in conda.anaconda.org huggingface.co cas-bridge.xethub.hf.co hf-mirror.com www.modelscope.cn files.pythonhosted.org download-r2.pytorch.org; do
  if curl -sS -m 10 -o /dev/null -I "https://$h/" 2> "$TMPDIR/net"; then
    echo "  ok       $h"
  else
    echo "  BLOCKED  $h  ($(tail -n 1 "$TMPDIR/net"))"
  fi
done

ENVP=$("$PY" -c 'import sys; print(sys.prefix)')
PYV=$("$PY" -c 'import sys, platform; print("%d.%d-%s" % (sys.version_info[:2] + (platform.machine(),)))')
CONDA="$PREFIX/miniforge/bin/conda"
say "Python packages (CPU build of PyTorch) -> $ENVP"
if "$PY" -c "import torch, transformers, peft, safetensors" > /dev/null 2>&1; then
  echo "  already installed"
elif [ -n "${WHEELS:-}" ]; then
  "$PY" -m pip install --no-index --find-links "$WHEELS" torch transformers peft safetensors accelerate
elif [ "$PYV" = 3.11-x86_64 ] && [ -d "$ENVP/conda-meta" ] && [ -x "$CONDA" ]; then
  # the flow's conda Python (tools/install_openroad_conda.sh): the tested conda-forge set (tools/lock/
  # llm-cpu-conda-py311.txt) from conda.anaconda.org -- the site the OpenROAD install already downloads from --
  # downloaded with curl (system certificates, like git), md5 checked, installed by conda offline
  PKG="$PREFIX/pkgs-llm"; mkdir -p "$PKG" "$PREFIX/pkgs-cache"; LIST="$PKG/explicit.txt"; echo "@EXPLICIT" > "$LIST"
  while read -r line; do
    url="${line%%#*}"; sum="${line##*#}"; f="$PKG/$(basename "$url")"
    if ! { [ -s "$f" ] && [ "$(md5sum < "$f" | cut -d' ' -f1)" = "$sum" ]; }; then
      echo "  download $(basename "$url")"
      curl -fL --retry 5 -C - -o "$f.part" "$url"
      mv "$f.part" "$f"
      [ "$(md5sum < "$f" | cut -d' ' -f1)" = "$sum" ] || { echo "  checksum mismatch, removed: $f (run again)"; rm -f "$f"; exit 1; }
    fi
    echo "file://$f#$sum" >> "$LIST"
  done < <(grep '^https://' "$REPO/tools/lock/llm-cpu-conda-py311.txt")
  [ ! -f "$PREFIX/condarc" ] || export CONDARC="$PREFIX/condarc"
  CONDA_PKGS_DIRS="$PREFIX/pkgs-cache" "$CONDA" install -y -q -p "$ENVP" --offline --file "$LIST"
  rm -rf "$PKG"
elif [ "$PYV" = 3.11-x86_64 ]; then
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
# every file is checked against tools/lock/qwen2.5-3b-instruct-bnb-4bit.sha256, so any source with the same bytes
# will do: MODEL_FROM (a folder you copied the files into), $HF_ENDPOINT (e.g. a company mirror), huggingface.co,
# hf-mirror.com, ModelScope; the first one that gives the right bytes is used
SUMS="$REPO/tools/lock/qwen2.5-3b-instruct-bnb-4bit.sha256"
okfile() { [ -s "$1" ] && [ "$(sha256sum < "$1" | cut -d' ' -f1)" = "$(awk -v f="$2" '$2 == f {print $1}' "$SUMS")" ]; }
SRCS=()
[ -z "${HF_ENDPOINT:-}" ] || SRCS+=("${HF_ENDPOINT%/}/unsloth/Qwen2.5-3B-Instruct-bnb-4bit/resolve/$REV")
SRCS+=("https://huggingface.co/unsloth/Qwen2.5-3B-Instruct-bnb-4bit/resolve/$REV"
       "https://hf-mirror.com/unsloth/Qwen2.5-3B-Instruct-bnb-4bit/resolve/$REV"
       "https://www.modelscope.cn/models/unsloth/Qwen2.5-3B-Instruct-bnb-4bit/resolve/master")
missing=()
for f in added_tokens.json config.json generation_config.json merges.txt special_tokens_map.json tokenizer.json \
         tokenizer_config.json vocab.json model.safetensors; do
  if okfile "$MODEL/$f" "$f"; then echo "  have $f"; continue; fi
  if [ -n "${MODEL_FROM:-}" ] && okfile "$MODEL_FROM/$f" "$f"; then
    echo "  copy $f from $MODEL_FROM"; cp "$MODEL_FROM/$f" "$MODEL/$f"; continue
  fi
  got=
  for k in "${!SRCS[@]}"; do
    case " ${dead:-} " in *" $k "*) continue ;; esac   # a source that failed once is not tried again
    s="${SRCS[$k]}"
    echo "  download $f from ${s#https://}"
    if [ "$f" = model.safetensors ]; then pr=--progress-bar; else pr=-sS; fi
    if curl -fL $pr --connect-timeout 15 --retry 2 -C - -o "$MODEL/$f.part$k" "$s/$f"; then
      if okfile "$MODEL/$f.part$k" "$f"; then mv "$MODEL/$f.part$k" "$MODEL/$f"; got=1; break; fi
      echo "    not the same file (sha256 differs): next source"; rm -f "$MODEL/$f.part$k"
    else
      echo "    failed: this source is skipped from now on"; dead="${dead:-} $k"
    fi
  done
  [ -n "$got" ] || missing+=("$f")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo
  echo "  the model could not be downloaded here (every source above is blocked or failed): ${missing[*]}"
  echo "  Get these files of unsloth/Qwen2.5-3B-Instruct-bnb-4bit (revision $REV) where you can, e.g."
  echo "    https://huggingface.co/unsloth/Qwen2.5-3B-Instruct-bnb-4bit/tree/$REV"
  echo "  put them in one folder on this machine, and run again with:  MODEL_FROM=/that/folder"
  echo "  (or copy them straight into $MODEL/). Their sha256 must match $SUMS."
  exit 1
fi
say "checksums"
(cd "$MODEL" && sha256sum -c "$REPO/tools/lock/qwen2.5-3b-instruct-bnb-4bit.sha256")
rm -rf "$TMPDIR"

say "self-test: load the model with the default adapter and score two prompts on the CPU"
LLM_DEVICE=cpu LLM_BASE_MODEL="$MODEL" "$PY" "$REPO/flow/lib/scorer.py" --selftest
echo
echo "done. In each new terminal: source $PREFIX/env.csh (bash: env.sh), then"
echo "  python3 flow/run.py A.def B.lef C.lef --policy llm"
