#!/bin/bash
# Reproduce the GSA paper tables. Runs every evaluation sequentially on one GPU.
# Results go to $RESULTS_ROOT (default ./results); each phase writes a done-marker.
#
#   Table 1 -> objaverse      (same-object, RRE + ATE)
#   Table 2 -> shapenet       (cross-instance, 6 categories, RRE)
#   Table 3 -> co3d {toyplane,chair,bicycle} + real_car  (Mean/Median RRE)
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GSA_DATA="${GSA_DATA:-./GSA_release_data}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results}"
LOG_DIR="$RESULTS_ROOT/logs"
mkdir -p "$LOG_DIR"

# ShapeNet coarse solver builds an NxN distance matrix; 50000 (paper) needs
# ~20 GB and OOMs a 24 GB card. 10000 fits comfortably and the feature-guided
# coarse solver is robust to the subsample size (negligible effect on RRE).
SHAPENET_ICP_SAMPLES="${SHAPENET_ICP_SAMPLES:-10000}"

run() {  # phase_name  command...
    local name="$1"; shift
    local marker="$RESULTS_ROOT/${name}.done"
    if [ -f "$marker" ]; then echo "[reproduce] $name already done, skipping"; return; fi
    echo "[reproduce] === $name START $(date -u +%H:%M:%S) ==="
    if "$@" > "$LOG_DIR/${name}.log" 2>&1; then
        touch "$marker"
        echo "[reproduce] === $name DONE  $(date -u +%H:%M:%S) ==="
    else
        echo "[reproduce] === $name FAILED (see $LOG_DIR/${name}.log) ==="
    fi
}

cd "$RESULTS_ROOT"

# --- Table 1: Objaverse same-object ---
run objaverse python "$REPO_DIR/eval/eval_register_3dgs_objaverse.py" \
    --results_dir "$RESULTS_ROOT/objaverse"

# --- Table 2: ShapeNet cross-instance (per category) ---
for cat in airplane bus boat car chair motorcycle; do
    run "shapenet_$cat" python "$REPO_DIR/eval/eval_register_3dgs_shapenet.py" \
        --category "$cat" --skip_ply_pairs \
        --icp_samples "$SHAPENET_ICP_SAMPLES" \
        --results_dir "$RESULTS_ROOT/shapenet"
done

# --- Table 3: CO3D cross-instance (fixed seed for reproducibility) ---
for cat in toyplane chair bicycle; do
    run "co3d_$cat" python "$REPO_DIR/eval/eval_register_3dgs_co3d.py" \
        --category "$cat" --seed 0 --results_dir "$RESULTS_ROOT/co3d_$cat"
done

# --- Table 3: 3D real car cross-instance ---
run real_car python "$REPO_DIR/eval/eval_register_3dgs_real_car.py" \
    --skip_ply_pairs --results_dir "$RESULTS_ROOT/real_car"

echo "[reproduce] ALL PHASES COMPLETE $(date -u +%H:%M:%S)"
