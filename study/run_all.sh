#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run all 5 study configs sequentially (one finishes before the next starts).
# Each config gets exclusive GPU access — identical to the original train.py.
# Existing run directories are backed up with a timestamp before overwriting.
#
# Usage (from repo root):
#   nohup bash study/run_all.sh > study/run_all.log 2>&1 &
#
# Monitor current run:
#   tail -f study/runs/cfg1/log.txt   (or whichever is active)
#
# Check overall progress:
#   cat study/run_all.log
# ─────────────────────────────────────────────────────────────────────────────

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STUDY_DIR="$REPO_ROOT/study"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

echo "========================================================"
echo "  AlphaZero Gomoku – Sequential Study Runs"
echo "  Repo: $REPO_ROOT"
echo "  Started: $(date)"
echo "========================================================"

CONFIGS=("cfg1" "cfg2" "cfg3" "cfg4" "cfg5")

for CFG in "${CONFIGS[@]}"; do
    RUN_DIR="$STUDY_DIR/runs/$CFG"
    CONFIG_FILE="$STUDY_DIR/configs/$CFG.json"

    # Back up existing run dir before overwriting
    if [ -d "$RUN_DIR" ]; then
        BACKUP="$STUDY_DIR/runs/${CFG}_backup_${TIMESTAMP}"
        mv "$RUN_DIR" "$BACKUP"
        echo "  Backed up existing $CFG → runs/${CFG}_backup_${TIMESTAMP}"
    fi
    mkdir -p "$RUN_DIR"

    echo ""
    echo "  ── Starting $CFG at $(date) ──"
    python "$STUDY_DIR/train_run.py" --config "$CONFIG_FILE"
    echo "  ── Finished $CFG at $(date) ──"
done

echo ""
echo "========================================================"
echo "  All 5 runs complete.  $(date)"
echo "========================================================"
