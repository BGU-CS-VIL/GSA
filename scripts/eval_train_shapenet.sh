#!/bin/bash
# run_shapenet_training.sh

# Data comes from the GSA release drop (override with GSA_DATA); retrained
# models go to a LOCAL directory by default (pass a different one as $1).
DATA_ROOT="${GSA_DATA:-./GSA_release_data}"
PARENT_DIR="$DATA_ROOT/shapenet/data"
OUTPUT_DIR="${1:-./retrained_models/shapenet}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Count total directories for progress bar
total_dirs=$(find "$PARENT_DIR" -mindepth 2 -maxdepth 2 -type d | wc -l)
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

# Process each category
for category_dir in "$PARENT_DIR"/*; do
    if [ -d "$category_dir" ]; then
        category_name=$(basename "$category_dir")
        echo -e "\nProcessing category: $category_name"
        
        # Create category output directory
        mkdir -p "$OUTPUT_DIR/$category_name"
        
        # Process each object in the category
        for object_dir in "$category_dir"/*; do
            if [ -d "$object_dir" ]; then
                object_id=$(basename "$object_dir")
                
                # Skip if directory is named "checkpoints"
                if [ "$object_id" = "checkpoints" ]; then
                    continue
                fi
                
                ((current++))
                print_progress $current $total_dirs
                
                echo -e "\nStarting ${category_name}/${object_id}"
                python "$REPO_DIR/train.py" \
                    -s "$object_dir" \
                    -m "$OUTPUT_DIR/$category_name/${object_id}" \
                    --iterations 7000 \
                    > "$OUTPUT_DIR/logs/${category_name}_${object_id}.log" 2>&1
                    
                echo -e "\nCompleted ${category_name}/${object_id}"
            fi
        done
    fi
done

echo -e "\nAll training jobs have been completed!"
echo "Check individual log files in logs/ directory for training logs"
