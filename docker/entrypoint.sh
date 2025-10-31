#!/bin/bash

if [ "$MODE" = "preprocess" ]; then
    echo "Starting in preprocessing mode..."
    python -u preprocess.py
elif [ "$MODE" = "training" ]; then
    echo "Starting in training mode..."
    python -u training.py
elif [ "$MODE" = "inference" ]; then
    echo "Starting in inference mode..."
    python -u interference.py
else
    echo "No mode specified. Available modes: preprocess, training, inference"
    echo "Set MODE environment variable to 'preprocess', 'training', or 'inference'"
    exit 1
fi