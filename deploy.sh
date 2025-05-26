#!/bin/bash
# Standard deployment script for TransMet
# Created by organization automation

echo "Deploying TransMet..."

# Check for requirements
if [ -f "requirements.txt" ]; then
  echo "Installing requirements..."
  pip install -r requirements.txt
fi

# Install the package in development mode
if [ -f "setup.py" ]; then
  echo "Installing package in development mode..."
  pip install -e .
fi

echo "Deployment complete!"
