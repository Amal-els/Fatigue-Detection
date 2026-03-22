#!/bin/bash
# Driver Fatigue Monitor - Setup Script for Linux/macOS

set -e

echo ""
echo "============================================================"
echo "Driver Fatigue Monitor - Setup Script"
echo "============================================================"
echo ""

cd "$(dirname "$0")"

# Run the Python setup script
python3 setup.py

echo ""
echo "============================================================"
echo "Setup complete! You can now run the app with:"
echo "  source .venv/bin/activate"
echo "  streamlit run fatigue_app.py"
echo "============================================================"
echo ""
