#!/usr/bin/env bash
# Build dist/polero-src.zip for a Kaggle Dataset (T2.5): code, configs, scripts, data and the
# fixed folds file, taken from the working tree (tracked + new non-ignored files).
# VERSION records the commit; "+dirty" means the zip contains uncommitted changes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=${PY:-.venv/bin/python}
OUT=${1:-dist/polero-src.zip}
mkdir -p "$(dirname "$OUT")"
[ -f artifacts/folds.csv ] || "$PY" -m polero.folds
PATHS=(src configs scripts pyproject.toml requirements.txt train_data.csv test_data.csv sample_output.csv)
DIRTY=""
git diff --quiet HEAD -- "${PATHS[@]}" && [ -z "$(git ls-files -o --exclude-standard -- "${PATHS[@]}")" ] || DIRTY="+dirty"
VERSION="$(git rev-parse HEAD)$DIRTY $(git rev-parse --abbrev-ref HEAD)"
git ls-files -co --exclude-standard -- "${PATHS[@]}" | grep -v '__pycache__' > /tmp/polero_pkg_files.$$
echo artifacts/folds.csv >> /tmp/polero_pkg_files.$$
"$PY" - "$OUT" "$VERSION" /tmp/polero_pkg_files.$$ <<'EOF'
import sys, zipfile
out, version, listing = sys.argv[1:]
files = [l for l in open(listing).read().splitlines() if l]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f)
    z.writestr("VERSION", version + "\n")
print(f"wrote {out}: {len(files)} files, version {version}")
EOF
rm -f /tmp/polero_pkg_files.$$
