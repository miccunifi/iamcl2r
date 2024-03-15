#!/bin/bash

#check if venv exists  
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

echo "Installing dependencies..."
pip install -e .
echo "Installation completed."

echo
echo
echo "To run the repo, use the following command:"
iamcl2r --help