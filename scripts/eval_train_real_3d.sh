#!/bin/bash
# run_3d_real_car_training.sh

# Data comes from the GSA release drop (override with GSA_DATA); retrained
# models go to a LOCAL directory by default (pass a different one as $1).
#
# NOTE: the released real-car models were manually aligned to the reference
# scan 2024_04_23_15_41_09 to define ground truth. Retraining from scratch
# produces models WITHOUT that alignment — use the released prebuilt models
# for evaluation. This script only documents how the models were built.
DATA_ROOT="${GSA_DATA:-./GSA_release_data}"
PARENT_DIR="$DATA_ROOT/real_car/data"
OUTPUT_DIR="${1:-./retrained_models/real_car}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Count total directories for progress bar
total_dirs=$(find "$PARENT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
current=0

# Create logs directory
mkdir -p "$OUTPUT_DIR/logs"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to print progress bar
print_progress() {
    local current=$1
    local total=$2
    local progress=$((current * 100 / total))
    local completed=$((progress / 2))
    local remaining=$((50 - completed))
    
    printf "\rProgress: [%-${completed}s%-${remaining}s] %d/%d (%d%%)" \
           "$(printf '%0.s=' {1..50})" \
           "$(printf '%0.s ' {1..50})" \
           "$current" "$total" "$progress"
}

echo "Total object directories found: $total_dirs"
echo "Starting training processes..."

# Process each object in the eval directory
for object_dir in "$PARENT_DIR"/*; do
    if [ -d "$object_dir" ]; then
        object_id=$(basename "$object_dir")
        
        ((current++))
        print_progress $current $total_dirs
        
        echo -e "\nStarting training for: $object_id"
        
        python "$REPO_DIR/train.py" \
            -s "$object_dir" \
            -m "$OUTPUT_DIR/${object_id}" \
            --iterations 7000 \
            > "$OUTPUT_DIR/logs/${object_id}.log" 2>&1
            
        echo -e "\nCompleted training for: $object_id"
    fi

done

echo -e "\nAll training jobs have been completed!"
echo "Check individual log files in logs/ directory for training logs"
