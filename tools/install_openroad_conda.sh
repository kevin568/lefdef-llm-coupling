#!/bin/bash
# Install OpenROAD + the flow's Python into one folder, without root, for Linux machines where the OpenROAD
# Ubuntu packages do not run (e.g. RHEL/Rocky 8 with glibc 2.28) and no Docker/Apptainer is available.
#
#   bash tools/install_openroad_conda.sh [FOLDER]   # default FOLDER: local-tools/ inside this repository folder
#   source local-tools/env.sh                       # each new terminal; then: python flow/run.py A.def B.lef --policy heuristic
#
# Nothing is written outside FOLDER (conda's cache, temporary files, config and home-directory files are redirected
# into it). About 3 GB while installing, 1.9 GB after. local-tools/ is in .gitignore. Running again resumes.
#
# Downloads only, with curl (the same way git connects): Miniforge (github.com/conda-forge/miniforge) and the
# packages listed in tools/lock/ (conda-forge, plus OpenROAD 2.0-12381 from litex-hub; the set tested end to end).
# conda installs from those local files offline. Every library OpenROAD loads is inside FOLDER (glibc 2.17+).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DEST="${1:-$REPO/local-tools}"
mkdir -p "$DEST"
PREFIX=$(cd "$DEST" && pwd)
CONDA="$PREFIX/miniforge/bin/conda"
OR_ENV="$PREFIX/env-openroad"; PY_ENV="$PREFIX/env-python"; PKG="$PREFIX/pkgs-download"
say() { printf '\n==> %s\n' "$*"; }
fetch() { curl -fL --retry 3 -o "$2" "$1"; }

# keep every write inside PREFIX
export CONDA_PKGS_DIRS="$PREFIX/pkgs-cache" TMPDIR="$PREFIX/tmp" CONDARC="$PREFIX/condarc"
export XDG_CACHE_HOME="$PREFIX/home/.cache" XDG_CONFIG_HOME="$PREFIX/home/.config"
mkdir -p "$CONDA_PKGS_DIRS" "$TMPDIR" "$PREFIX/home" "$PKG"
printf 'pkgs_dirs:\n  - %s\nchannels:\n  - conda-forge\nauto_update_conda: false\nnotify_outdated_conda: false\n' \
  "$CONDA_PKGS_DIRS" > "$CONDARC"
cnd() { HOME="$PREFIX/home" "$CONDA" "$@"; }

if [ ! -x "$CONDA" ]; then
  say "Miniforge -> $PREFIX/miniforge"
  fetch "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" "$PREFIX/miniforge.sh"
  HOME="$PREFIX/home" bash "$PREFIX/miniforge.sh" -b -u -p "$PREFIX/miniforge" > "$PREFIX/miniforge.log"   # -u: resume
  rm -f "$PREFIX/miniforge.sh"
fi

install_lock() {   # $1 environment folder, $2 lock file: curl downloads the packages, conda installs them offline
  local env="$1" lock="$2" list="$PKG/local_$(basename "$2")" url f
  echo "@EXPLICIT" > "$list"
  while read -r line; do
    url="${line%%#*}"; f="$PKG/$(basename "$url")"
    [ -s "$f" ] || { echo "  download $(basename "$url")"; fetch "$url" "$f"; }
    echo "file://$f" >> "$list"
  done < <(grep '^https://' "$lock")
  cnd create -y -q -p "$env" --offline --file "$list"
}

say "OpenROAD 2.0-12381 and its libraries -> $OR_ENV"
install_lock "$OR_ENV" "$REPO/tools/lock/env-openroad.txt"

say "older library versions this OpenROAD build links (fmt 8, ICU 64, libjpeg 9)"
for u in https://conda.anaconda.org/conda-forge/linux-64/fmt-8.1.1-h4bd325d_0.tar.bz2 \
         https://conda.anaconda.org/conda-forge/linux-64/icu-64.2-he1b5a44_1.tar.bz2 \
         https://conda.anaconda.org/conda-forge/linux-64/jpeg-9e-h7f98852_0.tar.bz2; do
  f="$PKG/$(basename $u)"; [ -s "$f" ] || fetch "$u" "$f"
  d="$PKG/x_$(basename $u .tar.bz2)"; mkdir -p "$d"; tar xjf "$f" -C "$d"
  for so in "$d"/lib/lib*.so.*; do [ -e "$OR_ENV/lib/$(basename $so)" ] || cp -a "$so" "$OR_ENV/lib/"; done
done
[ -e "$OR_ENV/lib/libgomp.so.1" ] || ln -s "$(ls "$OR_ENV"/lib/libgomp.so.1.* | head -1)" "$OR_ENV/lib/libgomp.so.1"

say "Python 3.11 for the flow (numpy, scipy) -> $PY_ENV"
install_lock "$PY_ENV" "$REPO/tools/lock/env-python.txt"

cat > "$PREFIX/env.sh" <<EOF
# source me: OpenROAD and the flow's Python
export OPENROAD="$OR_ENV/bin/openroad"
export PATH="$PY_ENV/bin:\$PATH"
EOF

cnd clean -afy -q > /dev/null; rm -rf "$PKG" "$TMPDIR"   # downloaded packages are no longer needed

say "check"
missing=$(env -i PATH=/usr/bin:/bin ldd "$OR_ENV/bin/openroad" | grep "not found" || true)
[ -z "$missing" ] || { echo "missing libraries:"; echo "$missing"; exit 1; }
"$OR_ENV/bin/openroad" -version
"$PY_ENV/bin/python" -c "import numpy, scipy, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
du -sh "$PREFIX" | sed 's/^/size: /'
echo
echo "done. Each new terminal:  source $PREFIX/env.sh"
echo "      then, in $REPO:  python flow/run.py A.def B.lef --policy heuristic"
