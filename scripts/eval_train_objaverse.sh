#!/bin/bash
# run_training.sh

# Data comes from the GSA release drop (override with GSA_DATA); retrained
# models go to a LOCAL directory by default (pass a different one as $1).
DATA_ROOT="${GSA_DATA:-./GSA_release_data}"
PARENT_DIR="$DATA_ROOT/objaverse/data"
OUTPUT_DIR="${1:-./retrained_models/objaverse}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Count total directories for progress bar
total_dirs=$(ls -d "$PARENT_DIR"/*/ 2>/dev/null | wc -l)
total_jobs=$((total_dirs * 2)) # Each dir has 2 jobs (block0 and block1)
current=0

# Create logs directory
mkdir -p "$OUTPUT_DIR/logs"

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

echo "Total directories found: $total_dirs"
echo "Total jobs to run: $total_jobs"
echo "Starting training processes..."

# Find all subdirectories
for data_dir in "$PARENT_DIR"/*; do
    if [ -d "$data_dir" ]; then
        dir_name=$(basename "$data_dir")
        
        # Process block0
        ((current++))
        print_progress $current $total_jobs
        
        echo -e "\nStarting ${dir_name} block0"
        python "$REPO_DIR/train.py" \
            -s "$data_dir/block0" \
            -m "$OUTPUT_DIR/${dir_name}_block0" \
            --iterations 7000 \
            > "$OUTPUT_DIR/logs/${dir_name}_block0.log" 2>&1
            
        echo -e "\nCompleted ${dir_name} block0"

        # Process block1
        ((current++))
        print_progress $current $total_jobs
        
        echo -e "\nStarting ${dir_name} block1"
        python "$REPO_DIR/train.py" \
            -s "$data_dir/block1" \
            -m "$OUTPUT_DIR/${dir_name}_block1" \
            --iterations 7000 \
            > "$OUTPUT_DIR/logs/${dir_name}_block1.log" 2>&1
            
        echo -e "\nCompleted ${dir_name} block1"
    fi
done

echo -e "\nAll training jobs have been completed!"
echo "Check individual log files in logs/ directory for training logs"