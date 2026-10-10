#!/usr/bin/env bash
# clean_cache.sh — tidy ExoSphere's data_cache/ and scratch outputs.
#
# Caches and what they hold:
#   data_cache/benchmark/       pinned WASP-39 b spectra + SHA256 manifest (KEEP: source data)
#   data_cache/exoarchive/      TAP CSV responses + downloaded spectra (KEEP: archive cache)
#   data_cache/mast/            MAST manifests/products (KEEP: archive cache)
#   data_cache/ml/              synthetic training dataset + metadata (KEEP: hours to rebuild)
#   models/                     trained checkpoints (KEEP)
#   data_cache/cleaned/         per-analysis cleaned spectra (regenerable)
#   data_cache/posteriors/      per-analysis posterior npz (regenerable, slow)
#   data_cache/preprocess_logs/ per-analysis removal logs (regenerable)
#   data_cache/checkpoints/     dynesty resume state (regenerable)
#   data_cache/reports/         built HTML reports (regenerable)
#   outputs/                    run logs, scratch scripts, L2/L3 markdown (regenerable)
#   exosphere.db                dev SQLite database (regenerable via API/pipeline)
#
# Usage:
#   ./scripts/clean_cache.sh [--dry-run] [--all] [--db]
#     default : remove regenerable run products only
#     --all   : also remove ml dataset/models and archive caches (keeps benchmark)
#     --db    : also delete the dev database (you must re-seed/re-run analyses)
set -euo pipefail

DRY_RUN=0
ALL=0
DROP_DB=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --all) ALL=1 ;;
    --db) DROP_DB=1 ;;
    *) echo "unknown flag: $arg (see header comments)"; exit 1 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

remove_path() {
  if [ ! -e "$1" ]; then
    return
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "would remove: $1"
  else
    echo "removing: $1"
    rm -rf "$1"
  fi
}

# Regenerable run products (safe default).
for dir in \
  data_cache/cleaned \
  data_cache/posteriors \
  data_cache/preprocess_logs \
  data_cache/checkpoints \
  data_cache/reports \
  outputs \
  .pytest_cache \
  .ruff_cache \
; do
  remove_path "$dir"
done
find . -type d -name "__pycache__" -not -path "./.venv/*" -not -path "./web/*" | while read -r dir; do
  remove_path "$dir"
done

if [ "$ALL" -eq 1 ]; then
  for dir in data_cache/exoarchive data_cache/mast data_cache/ml models; do
    remove_path "$dir"
  done
fi

if [ "$DROP_DB" -eq 1 ]; then
  remove_path "exosphere.db"
fi

echo "done."
