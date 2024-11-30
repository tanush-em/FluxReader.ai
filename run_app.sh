#!/bin/bash

# Set up Python virtual environment
if [ ! -d "env-fluxreaderai" ]; then
    echo "Creating virtual environment..."
    python3 -m venv env-fluxreaderai
fi

# Activate virtual environment
echo "Activating virtual environment..."
source env-fluxreaderai/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the Streamlit application
echo "Starting the application..."
streamlit run app.py

# Deactivate virtual environment on exit
echo "Deactivating virtual environment..."
deactivate

