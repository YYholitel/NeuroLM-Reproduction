#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python "$ROOT/scripts/drive_official_tusl3.py" 2>&1 | tee "$ROOT/driver.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$ROOT/driver_exit_code.txt"
exit "$rc"
