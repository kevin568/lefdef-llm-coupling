#!/bin/bash
# Install OpenROAD + the flow's Python into one folder, without root, for Linux machines where the OpenROAD
# Ubuntu packages do not run (e.g. RHEL/Rocky 8 with glibc 2.28) and no Docker/Apptainer is available.
#
#   bash tools/install_openroad_conda.sh [FOLDER]   # default FOLDER: local-tools/ inside this repository folder
#   source local-tools/env.sh                       # each new terminal; then: python flow/run.py A.def B.lef --policy heuristic
#
# Nothing is written outside FOLDER: conda's package cache, temporary files, its config and its home-directory
# files (~/.conda, ~/.condarc, ~/.cache) are all redirected into FOLDER. About 3 GB while installing, 1.9 GB after.
# local-tools/ is in .gitignore, so `git pull` does not touch it. Running the script again resumes or repairs.
#
# Downloads only (nothing is uploaded): Miniforge (conda-forge's installer, github.com/conda-forge/miniforge),
# packages from conda-forge, and the OpenROAD build 2.0-12381 (Feb 2024) from the litex-hub channel. Anaconda's
# "defaults" channel is not used. Every library the binaries load is inside FOLDER (needs glibc 2.17 or newer).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DEST="${1:-$REPO/local-tools}"
if ! mkdir -p "$DEST" 2>/dev/null; then
  echo "ERROR: cannot create $DEST. Give a folder you can write to: bash $0 /path/you/can/write/local-tools"; exit 1
fi
PREFIX=$(cd "$DEST" && pwd)
CONDA="$PREFIX/miniforge/bin/conda"
OR_ENV="$PREFIX/env-openroad"; PY_ENV="$PREFIX/env-python"; PKG="$PREFIX/pkgs-download"
say() { printf '\n==> %s\n' "$*"; }
fetch() { curl -fsSL --retry 3 -o "$2" "$1"; }

# every write stays inside PREFIX: package cache, temporary files, config, conda's ~/.conda and ~/.cache files
export CONDA_PKGS_DIRS="$PREFIX/pkgs-cache" TMPDIR="$PREFIX/tmp" CONDARC="$PREFIX/condarc"
export XDG_CACHE_HOME="$PREFIX/home/.cache" XDG_CONFIG_HOME="$PREFIX/home/.config"
mkdir -p "$CONDA_PKGS_DIRS" "$TMPDIR" "$PREFIX/home"
printf 'pkgs_dirs:\n  - %s\nchannels:\n  - conda-forge\nauto_update_conda: false\nnotify_outdated_conda: false\n' \
  "$CONDA_PKGS_DIRS" > "$CONDARC"
cnd() { HOME="$PREFIX/home" "$CONDA" "$@"; }   # conda never sees the real home directory

if ! ( : > "$PREFIX/.write_test" ) 2>/dev/null; then
  echo "ERROR: cannot write in $PREFIX. Give a folder you can write to: bash $0 /path/you/can/write/local-tools"; exit 1
fi
rm -f "$PREFIX/.write_test"
need_kb=$((3 * 1024 * 1024)); have_kb=$(df -Pk "$PREFIX" | awk 'NR==2 {print $4}')
if [ "${have_kb:-0}" -lt "$need_kb" ]; then
  echo "WARNING: $PREFIX has $((${have_kb:-0} / 1024)) MB free; a fresh install needs about 3 GB while it runs."
  echo "         If it stops with 'write failed', free space there (check: df -h . ; quota -s) and run again."
fi

if [ ! -x "$CONDA" ]; then
  say "Miniforge -> $PREFIX/miniforge"
  fetch "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" "$PREFIX/miniforge.sh"
  HOME="$PREFIX/home" bash "$PREFIX/miniforge.sh" -b -u -p "$PREFIX/miniforge" > "$PREFIX/miniforge.log"   # -u: resume
  rm -f "$PREFIX/miniforge.sh"
fi

say "OpenROAD runtime libraries (conda-forge) -> $OR_ENV"
cnd create -y -q -p "$OR_ENV" --override-channels -c conda-forge \
  python=3.10 "boost-cpp=1.73" "spdlog=1.9.2" tk zlib libgomp xorg-libx11 libgl libpng libglib

say "OpenROAD 2.0-12381 (litex-hub) and Qt 5.9.7 (conda-forge), installed from the package files"
mkdir -p "$PKG"
OR_URL=https://conda.anaconda.org/litex-hub/linux-64/openroad-2.0_12381_g01bba3695-20240223_100318_py310.tar.bz2
QT_URL=https://conda.anaconda.org/conda-forge/linux-64/qt-5.9.7-h0c104cb_3.tar.bz2
for u in "$OR_URL" "$QT_URL"; do [ -s "$PKG/$(basename $u)" ] || fetch "$u" "$PKG/$(basename $u)"; done
# --no-deps from files: the OpenROAD package asks for "defaults"-channel names; conda-forge supplies the same libraries
cnd install -y -q -p "$OR_ENV" --offline --no-deps "$PKG/$(basename $OR_URL)" "$PKG/$(basename $QT_URL)"

say "older library versions this build links (fmt 8, ICU 64, libjpeg 9; their file names do not clash)"
for u in https://conda.anaconda.org/conda-forge/linux-64/fmt-8.1.1-h4bd325d_0.tar.bz2 \
         https://conda.anaconda.org/conda-forge/linux-64/icu-64.2-he1b5a44_1.tar.bz2 \
         https://conda.anaconda.org/conda-forge/linux-64/jpeg-9e-h7f98852_0.tar.bz2; do
  f="$PKG/$(basename $u)"; [ -s "$f" ] || fetch "$u" "$f"
  d="$PKG/x_$(basename $u .tar.bz2)"; mkdir -p "$d"; tar xjf "$f" -C "$d"
  for so in "$d"/lib/lib*.so.*; do [ -e "$OR_ENV/lib/$(basename $so)" ] || cp -a "$so" "$OR_ENV/lib/"; done
done
[ -e "$OR_ENV/lib/libgomp.so.1" ] || ln -s "$(ls "$OR_ENV"/lib/libgomp.so.1.* | head -1)" "$OR_ENV/lib/libgomp.so.1"

say "Python for the flow (conda-forge: numpy, scipy) -> $PY_ENV"
cnd create -y -q -p "$PY_ENV" --override-channels -c conda-forge python=3.11 numpy scipy

cat > "$PREFIX/env.sh" <<EOF
# source me: OpenROAD and the flow's Python (nothing outside $PREFIX is used)
export OPENROAD="$OR_ENV/bin/openroad"
export PATH="$PY_ENV/bin:\$PATH"
EOF

cnd clean -afy -q > /dev/null; rm -rf "$PKG" "$TMPDIR"   # downloaded package caches are no longer needed

say "self-test"
missing=$(env -i PATH=/usr/bin:/bin ldd "$OR_ENV/bin/openroad" | grep "not found" || true)
[ -z "$missing" ] || { echo "missing libraries:"; echo "$missing"; exit 1; }
"$OR_ENV/bin/openroad" -version
"$PY_ENV/bin/python" -c "import numpy, scipy, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
outside=$(env -i PATH=/usr/bin:/bin ldd "$OR_ENV/bin/openroad" | grep -v "$OR_ENV" | grep -v -E 'vdso|ld-linux|lib(c|m|dl|pthread|rt|util)\.so' || true)
[ -z "$outside" ] || { echo "note: these libraries come from the system:"; echo "$outside"; }
du -sh "$PREFIX" | sed 's/^/size: /'
echo
echo "done. Each new terminal:  source $PREFIX/env.sh"
echo "      then, in $REPO:  python flow/run.py A.def B.lef --policy heuristic"
