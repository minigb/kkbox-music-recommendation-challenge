#!/bin/bash

# List of Python files to run
python_files=("main_lgbm.py" "main_predict.py")

# Loop through the list and run each file
for file in "${python_files[@]}"; do
    python3 "$file"
    if [ $? -ne 0 ]; then
        echo "$file failed. Exiting."
        exit 1
    fi
done