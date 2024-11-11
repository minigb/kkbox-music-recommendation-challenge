#!/bin/bash

set -e

# Set the directory name where the Python scripts are located
dir_name="dataset_crawling"

# List of Python scripts to run
scripts=(
    "refine_given_dataset.py"
    "crawling_data.py"
)

# Iterate over the scripts and run each one
for script in "${scripts[@]}"
do
    if [ -f "$dir_name/$script" ]; then
        echo "Running $script"
        python "$dir_name/$script"
    else
        echo "Error: $script not found in $dir_name"
        exit 1
    fi
done

echo "All scripts have been executed successfully."