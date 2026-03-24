#!/bin/bash

echo "======================================"
echo " Preparing Sigma Male Detector"
echo "======================================"

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "Installing CV packages..."
pip install -r requirements.txt --quiet

echo "Launching..."
python sigma_detector.py

deactivate
