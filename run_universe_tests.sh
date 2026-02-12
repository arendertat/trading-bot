#!/bin/bash
# Run universe selection tests
# Assumes dependencies are installed in current environment

set -e

echo "Running universe selection tests..."
echo ""

# Try to find pytest
if command -v pytest &> /dev/null; then
    PYTEST_CMD="pytest"
elif python3 -m pytest --version &> /dev/null 2>&1; then
    PYTEST_CMD="python3 -m pytest"
else
    echo "ERROR: pytest not found. Please install dependencies:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Run tests
$PYTEST_CMD tests/test_universe.py -v --tb=short

echo ""
echo "Tests complete!"
