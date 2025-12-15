#!/usr/bin/env bash

echo "========================================"
echo " ROB422 PROJECT - Environment Installer"
echo "========================================"

# Name of the environment from env.yml
ENV_NAME="rob422_project"

# ----------------------------------------
# Check for conda
# ----------------------------------------
if ! command -v conda &> /dev/null
then
    echo "❌ Conda not found! Please install Miniconda or Anaconda first."
    exit 1
fi

echo "✔ Conda detected."

# ----------------------------------------
# Check env.yml exists
# ----------------------------------------
if [ ! -f "env.yml" ]; then
    echo "❌ env.yml not found in current directory."
    echo "Make sure install.sh and env.yml are in the same folder."
    exit 1
fi

echo "✔ env.yml found."

# # ----------------------------------------
# # Remove existing environment (if exists)
# # ----------------------------------------
# if conda info --envs | grep -q "$ENV_NAME"; then
#     echo "⚠ Environment '$ENV_NAME' already exists."
#     echo "Deleting old environment..."
#     conda env remove -n "$ENV_NAME" -y
# fi

# ----------------------------------------
# Create environment
# ----------------------------------------
echo "Creating conda environment from env.yml..."
conda env create -f env.yml

if [ $? -ne 0 ]; then
    echo "❌ Failed to create environment."
    exit 1
fi

echo "✔ Environment created successfully."

# ----------------------------------------
# Final instructions
# ----------------------------------------
echo ""
echo "========================================"
echo " INSTALLATION COMPLETE!"
echo "========================================"
echo "Activate your environment using:"
echo ""
echo "    conda activate $ENV_NAME"
echo ""
echo "Then run the project normally:"
echo ""
echo "    python demo.py"
echo ""
echo "========================================"
