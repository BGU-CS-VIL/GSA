#!/bin/bash
# Train a feature-3DGS model for a single object.
#
# Usage: scripts/train_object.sh SOURCE_DIR OUTPUT_DIR [PORT]
#
# SOURCE_DIR must contain:
#   - the object's images (train/ for Blender-style data with transforms_train.json,
#     or images/ + sparse/ for COLMAP data)
#   - spherical_feature_maps/ with the *_sphere_fmap_CxHxW.pt files produced by
#     sphere_extractor/extract_features.py
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 SOURCE_DIR OUTPUT_DIR [PORT]" >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$REPO_DIR/train.py" -s "$1" -m "$2" --iterations 7000 --port "${3:-6009}"
