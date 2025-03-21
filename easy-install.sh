#!/bin/bash
# Exit the script if any error occurs
set -e

# 1. Check if the "uv" command is installed; if not, install it using the curl command
if ! command -v uv &> /dev/null; then
    echo "uv command not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the uv environment variables
    source $HOME/.local/bin/env
fi

# 2. Create a virtual environment "venv" using uv with Python 3.12
echo "Creating the virtual environment 'venv' with Python 3.12 using uv..."
uv venv venv --python 3.12

# 3. Activate the virtual environment
echo "Activating the virtual environment 'venv'..."
source venv/bin/activate

# 4. Install packages using uv pip
echo "Installing packages using uv pip..."
uv pip install ninja wheel setuptools
uv pip install torch
uv pip install --no-build-isolation flash-attn
uv pip install -r requirements.txt
uv pip install -U "huggingface_hub[cli]" --system

# 5. Install .NET 9
echo "Downloading and installing .NET 9..."
wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh
chmod +x ./dotnet-install.sh
./dotnet-install.sh --version latest --runtime aspnetcore --channel 9.0

# 6. Set environment variables for .NET
export DOTNET_ROOT=$HOME/.dotnet
export PATH=$PATH:$DOTNET_ROOT

# 7. Change directory to ui/deploy and run the .NET application
echo "Navigating to ui/deploy and starting the .NET application..."
cd ui/deploy/
dotnet DiffusionPipeInterface.dll --urls=http://0.0.0.0:5000
