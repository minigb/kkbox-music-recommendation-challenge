#!/bin/bash

# Run main_lgbm.py
python3 main_lgbm.py

# Check if the previous command was successful
if [ $? -eq 0 ]; then
    # Run main_predict.py
    python3 main_predict.py
else
    echo "main_lgbm.py failed. Exiting."
    exit 1
fi